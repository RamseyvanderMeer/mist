"""
Validation pipeline for MIST scraped items.

Reuses logic from scripts/process_scraped_data.py for fault code validation,
OBD normalization, and quality scoring.
"""
import logging
from typing import Any

from scrapy import Spider
from scrapy.exceptions import DropItem

from scrapers.items import MistScrapedItem
from scrapers.utils.constants import FAULT_CODE_VALIDATE_PATTERNS, OBD_RANGES

logger = logging.getLogger(__name__)


def validate_fault_code(code: str) -> bool:
    """Validate fault code format."""
    if not code or not isinstance(code, str):
        return False
    code = code.strip().upper()
    return any(p.match(code) for p in FAULT_CODE_VALIDATE_PATTERNS)


def normalize_obd_data(obd_data: dict[str, Any]) -> dict[str, float]:
    """Normalize and validate OBD-II data."""
    if not obd_data or not isinstance(obd_data, dict):
        return {}
    normalized = {}
    for key, value in obd_data.items():
        key_lower = key.lower().replace(" ", "_").replace("-", "_")
        try:
            float_value = float(value)
            if key_lower in OBD_RANGES:
                min_val, max_val = OBD_RANGES[key_lower]
                if min_val <= float_value <= max_val:
                    normalized[key_lower] = float_value
            else:
                normalized[key_lower] = float_value
        except (ValueError, TypeError):
            continue
    return normalized


# Action verbs and concrete repair terms that indicate a real solution
SOLUTION_ACTION_PATTERNS = (
    "replaced", "installed", "fixed", "repaired", "cleaned", "rebuilt",
    "adjusted", "tightened", "sealed", "flushed", "changed", "swapped",
    "upgraded", "reset", "cleared", "diagnosed", "tested", "removed",
)


def _solution_is_substantial(repair_summary: str) -> bool:
    """
    True if repair summary describes a concrete fix, not vague/generic.
    Rejects: "fixed it", "it worked", "replaced something".
    """
    if not repair_summary or len(repair_summary) < 50:
        return False
    s = repair_summary.lower()
    # Must mention a concrete action or part
    if any(term in s for term in SOLUTION_ACTION_PATTERNS):
        return True
    # Reject very vague
    vague = ("fixed it", "it worked", "that fixed it", "problem solved", "all good")
    if any(v in s for v in vague) and len(s) < 80:
        return False
    # Accept if substantial length and mentions common repair words
    if len(s) >= 80 and any(w in s for w in ("part", "sensor", "coil", "plug", "valve", "pump")):
        return True
    return len(s) >= 100  # Long enough to be substantive


def calculate_quality_score(item: MistScrapedItem, *, is_cause_to_solution: bool = False) -> float:
    """
    Calculate data quality score (0.0-1.0).

    Scoring (aligned with process_scraped_data.py):
    - Fault codes: 0.3 (or symptoms: 0.2 for cause_to_solution)
    - Vehicle context: 0.2
    - Repair summary: 0.3
    - OBD data: 0.15
    - Outcome: 0.05
    """
    score = 0.0

    fault_codes = item.get("fault_codes") or []
    if fault_codes and any(validate_fault_code(c) for c in fault_codes):
        score += 0.3
    elif is_cause_to_solution:
        symptoms = (item.get("symptoms") or "").strip()
        if len(symptoms) >= 20:
            score += 0.2
        elif len(symptoms) > 0:
            score += 0.1

    vehicle_context = item.get("vehicle_context") or {}
    if vehicle_context:
        has_make = bool(vehicle_context.get("make"))
        has_model = bool(vehicle_context.get("model"))
        has_year = bool(vehicle_context.get("year"))
        if has_make and has_model and has_year:
            score += 0.2
        elif (has_make and has_model) or (has_make and has_year):
            score += 0.1

    repair_summary = (item.get("repair_summary") or "").strip()
    if len(repair_summary) >= 50:
        score += 0.3
    elif len(repair_summary) > 0:
        score += 0.15

    obd_data = item.get("obd_data") or {}
    if obd_data:
        n = len(obd_data)
        if n >= 5:
            score += 0.15
        elif n >= 3:
            score += 0.1
        elif n >= 1:
            score += 0.05

    outcome = (item.get("outcome") or "").lower()
    if outcome in ("success", "failure", "partial"):
        score += 0.05

    return score


class MistValidationPipeline:
    """Validate and filter MistScrapedItem by quality score."""

    def __init__(self, min_quality: float = 0.6):
        self.min_quality = min_quality

    @classmethod
    def from_crawler(cls, crawler):
        return cls(min_quality=crawler.settings.getfloat("MIST_MIN_QUALITY", 0.6))

    def process_item(self, item, spider: Spider):
        # Pass through items that don't have fault_codes (not from our LLM pipeline)
        if "fault_codes" not in item:
            return item
        if item.get("fault_codes") is None:
            item["fault_codes"] = []

        fault_codes = item.get("fault_codes") or []
        valid_codes = [
            c.strip().upper()
            for c in fault_codes
            if validate_fault_code(c)
        ]

        repair_summary = (item.get("repair_summary") or "").strip()
        outcome = (item.get("outcome") or "unknown").lower()
        symptoms = (item.get("symptoms") or "").strip()

        # Path 1: Has valid fault codes → fault_code record
        if valid_codes:
            item["fault_codes"] = valid_codes
            item["record_type"] = "fault_code"
        # Path 2: No fault codes but has cause-to-solution (symptoms + confirmed fix)
        elif repair_summary and _solution_is_substantial(repair_summary):
            if outcome in ("success", "partial") or symptoms:
                item["fault_codes"] = []
                item["record_type"] = "cause_to_solution"
                if symptoms:
                    item["symptoms"] = symptoms
            else:
                raise DropItem("Cause-to-solution requires outcome or symptoms")
        else:
            raise DropItem("No valid fault codes and no substantial cause-to-solution")

        # Validate and normalize
        item["obd_data"] = normalize_obd_data(item.get("obd_data") or {})

        is_cause_to_solution = item.get("record_type") == "cause_to_solution"
        heuristic_score = calculate_quality_score(item, is_cause_to_solution=is_cause_to_solution)
        item["heuristic_score"] = heuristic_score

        llm_confidence = item.get("llm_confidence")

        if llm_confidence is not None:
            item["confidence_score"] = llm_confidence
            if heuristic_score < 0.3:
                raise DropItem(f"Heuristic quality score {heuristic_score:.2f} too low despite LLM confidence")
        else:
            item["confidence_score"] = heuristic_score
            # Cause-to-solution: lower threshold (0.5) since we validated solution quality
            min_q = 0.5 if is_cause_to_solution else self.min_quality
            if heuristic_score < min_q:
                raise DropItem(f"Quality score {heuristic_score:.2f} below threshold {min_q}")

        return item

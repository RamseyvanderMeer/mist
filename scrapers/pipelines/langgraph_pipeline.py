"""
LLM processing pipeline - Extraction with Gemini.

Analyzes forum content, extracts confirmed fixes, and filters by confidence.
"""
import hashlib
import json
import logging
import os
import re
from scrapy import Spider
from scrapy.exceptions import DropItem

from scrapers.utils.extractors import (
    extract_fault_codes,
    extract_obd_data,
    extract_outcome,
    extract_repair_summary,
    extract_vehicle_context,
)

logger = logging.getLogger(__name__)

# Regex patterns for pre-filter: skip LLM if content has no fault codes or repair-like text
REPAIR_LIKE_PATTERN = re.compile(
    r"\b(replaced|fixed|repaired|installed|cleaned|rebuilt|adjusted|sealed|flushed|"
    r"changed|swapped|reset|cleared|diagnosed|that fixed|fixed it|problem solved|"
    r"solution was|fix was|replacement of|installed new)\b",
    re.IGNORECASE,
)


def _content_worth_llm(content: str) -> bool:
    """
    True if content likely has extractable data (fault codes or repair description).
    Skip LLM when False to save tokens.
    """
    if not content or len(content) < 200:
        return False
    if extract_fault_codes(content):
        return True
    if REPAIR_LIKE_PATTERN.search(content):
        return True
    return False

EXTRACTION_PROMPT = """Analyze this automotive forum thread. Extract structured diagnostic data.

Rules:
- Identify the CONFIRMED FIX only. The fix must be explicitly stated by someone who solved the problem (e.g., "Yes, I have. It was X" or "Replaced X and that fixed it"). Do NOT include speculation, suggestions, or causes—only the actual fix that worked.
- Watch for sarcasm, jokes, or uncertain language. If the fix seems sarcastic or speculative, set confidence low.
- Extract fault codes (P-codes like P0300, manufacturer codes like 2A87) if mentioned.
- Extract symptoms (what the car was doing wrong) if no fault codes—e.g. "rough idle", "check engine light", "smoking from exhaust".
- Extract vehicle context (make, model, year, engine) if mentioned.
- Set outcome: "success" if a fix is confirmed, "failure" if no fix, "partial" if unclear.
- Set confidence (0.0-1.0): How confident are you that the fix is real and not sarcasm/speculation?

CRITICAL: Respond with ONLY a single valid JSON object. No markdown, no code blocks, no backticks, no trailing commas, no comments.
Required keys: fault_codes (array), symptoms (string), vehicle_context (object), repair_summary (string), outcome (string), confidence (number).

Example with fault codes:
{{"fault_codes": ["P0300"], "symptoms": "", "vehicle_context": {{"make": "BMW", "model": "335i", "year": 2011}}, "repair_summary": "Replaced crankcase vent valve.", "outcome": "success", "confidence": 0.9}}

Example with symptoms only (no codes):
{{"fault_codes": [], "symptoms": "Rough idle, check engine light flashing", "vehicle_context": {{"make": "BMW", "model": "335i"}}, "repair_summary": "Replaced ignition coil on cylinder 3.", "outcome": "success", "confidence": 0.85}}

If no confirmed fix exists: {{"fault_codes": [], "symptoms": "", "vehicle_context": {{}}, "repair_summary": "", "outcome": "unknown", "confidence": 0.0}}

Forum content:
---
{content}
---
"""


class LLMExtractionPipeline:
    """Process items through Gemini LLM for extraction and confidence filtering."""

    # Gemini 2.0 Flash pricing (per million tokens): input $0.10, output $0.40
    INPUT_PRICE_PER_1M = 0.10
    OUTPUT_PRICE_PER_1M = 0.40

    def __init__(
        self,
        enabled: bool = True,
        min_confidence: float = 0.8,
        api_key: str | None = None,
        prefilter: bool = True,
    ):
        self.enabled = enabled
        self.min_confidence = min_confidence
        self._api_key = api_key
        self._prefilter = prefilter
        self._seen_hashes: set[str] = set()
        self._model = None
        self._token_stats = {"input": 0, "output": 0, "calls": 0}

    @classmethod
    def from_crawler(cls, crawler):
        api_key = crawler.settings.get("GEMINI_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        return cls(
            enabled=crawler.settings.getbool("LLM_EXTRACTION_ENABLED", True),
            min_confidence=crawler.settings.getfloat("LLM_MIN_CONFIDENCE", 0.8),
            api_key=api_key,
            prefilter=crawler.settings.getbool("LLM_PREFILTER_ENABLED", True),
        )

    def open_spider(self, spider: Spider) -> None:
        self._seen_hashes.clear()
        self._token_stats = {"input": 0, "output": 0, "calls": 0}
        if self.enabled and self._api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("LLM extraction enabled (Gemini)")
            except Exception as e:
                logger.warning("LLM extraction disabled: %s", e)
                self._model = None
        else:
            if not self._api_key:
                logger.warning("GEMINI_API_KEY not set; LLM extraction disabled")
            self._model = None

    def process_item(self, item, spider: Spider):
        content = item.get("html", "") or item.get("raw_text", "")
        if not content:
            return item

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            raise DropItem("Exact Duplicate")
        self._seen_hashes.add(content_hash)

        if not self.enabled or not self._model:
            if "raw_text" in item:
                del item["raw_text"]
            return item

        # Pre-filter: skip LLM when content has no fault codes or repair-like text
        if self._prefilter and not _content_worth_llm(content):
            logger.debug("Skipping LLM (no fault codes or repair-like text in content)")
            return self._apply_regex_fallback(item, content)

        try:
            result = self._extract_with_llm(content)
        except Exception as e:
            logger.warning("LLM extraction failed, falling back to regex: %s", e)
            result = None

        if result:
            confidence = result.get("confidence", 0.0)
            repair_summary = (result.get("repair_summary") or "").strip()

            if repair_summary and confidence >= self.min_confidence:
                item["fault_codes"] = _normalize_fault_codes(result.get("fault_codes"))
                item["vehicle_context"] = _normalize_vehicle_context(
                    result.get("vehicle_context")
                )
                item["repair_summary"] = repair_summary
                item["outcome"] = result.get("outcome", "success")
                item["symptoms"] = (result.get("symptoms") or "").strip() or None
                item["llm_confidence"] = confidence
                item["confidence_score"] = confidence
                if "raw_text" in item:
                    del item["raw_text"]
                logger.debug(
                    "LLM extracted: %s (confidence %.2f)",
                    repair_summary[:80],
                    confidence,
                )
                return item

        # Fallback: regex extraction from raw text
        return self._apply_regex_fallback(item, content)

    def _apply_regex_fallback(self, item, content: str):
        """Populate item from regex extractors when LLM fails."""
        item["fault_codes"] = extract_fault_codes(content)
        item["obd_data"] = item.get("obd_data") or extract_obd_data(content)
        item["vehicle_context"] = item.get("vehicle_context") or extract_vehicle_context(
            content
        )
        if not (item.get("repair_summary") or "").strip():
            item["repair_summary"] = extract_repair_summary(content)
        if (item.get("outcome") or "unknown") == "unknown":
            item["outcome"] = extract_outcome(content)
        item["llm_confidence"] = None
        item["confidence_score"] = 0.5  # Heuristic fallback
        if "raw_text" in item:
            del item["raw_text"]
        logger.debug("Regex fallback: fault_codes=%s", item["fault_codes"])
        return item

    def _extract_with_llm(self, content: str) -> dict | None:
        prompt = EXTRACTION_PROMPT.format(content=content[:50000])
        response = self._model.generate_content(prompt)
        # Track token usage
        um = getattr(response, "usage_metadata", None)
        if um:
            inp = getattr(um, "prompt_token_count", 0) or 0
            out = getattr(um, "candidates_token_count", 0) or getattr(um, "output_token_count", 0) or 0
            self._token_stats["input"] += inp
            self._token_stats["output"] += out
            self._token_stats["calls"] += 1
        text = (response.text or "").strip()
        if not text:
            return None
        return _parse_llm_json(text)

    def close_spider(self, spider: Spider) -> None:
        """Log token usage and cost when spider closes."""
        if self._token_stats["calls"] > 0:
            inp, out, calls = (
                self._token_stats["input"],
                self._token_stats["output"],
                self._token_stats["calls"],
            )
            cost = (inp / 1_000_000 * self.INPUT_PRICE_PER_1M) + (
                out / 1_000_000 * self.OUTPUT_PRICE_PER_1M
            )
            logger.info(
                "LLM token usage: %d calls, %d input + %d output = %d total tokens, est. cost $%.6f",
                calls, inp, out, inp + out, cost,
            )


def _normalize_fault_codes(codes) -> list:
    """Ensure fault_codes is a list of valid strings."""
    if not codes:
        return []
    result = []
    for c in codes if isinstance(codes, list) else [codes]:
        if isinstance(c, str) and c.strip():
            result.append(c.strip().upper())
    return result


def _normalize_vehicle_context(ctx) -> dict:
    """Ensure vehicle_context is a dict with string values."""
    if not ctx or not isinstance(ctx, dict):
        return {}
    return {k: str(v).strip() for k, v in ctx.items() if v is not None}


def _parse_llm_json(text: str) -> dict | None:
    """
    Parse JSON from LLM response, tolerating markdown and common malformations.
    """
    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Extract JSON object (first complete {...})
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return None

    raw = json_match.group()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fix common LLM JSON issues
    fixed = raw
    # Remove trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: extract key fields with regex
    result = {}
    fc_match = re.search(r'"fault_codes"\s*:\s*\[([^\]]*)\]', raw)
    if fc_match:
        codes = re.findall(r'"([^"]+)"', fc_match.group(1))
        result["fault_codes"] = [c.strip().upper() for c in codes if c.strip()]
    else:
        result["fault_codes"] = []

    sym_match = re.search(r'"symptoms"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    result["symptoms"] = (sym_match.group(1) if sym_match else "").replace("\\n", " ").strip()

    vc_match = re.search(r'"vehicle_context"\s*:\s*(\{[^}]*\})', raw)
    if vc_match:
        try:
            result["vehicle_context"] = json.loads(vc_match.group(1))
        except json.JSONDecodeError:
            result["vehicle_context"] = {}
    else:
        result["vehicle_context"] = {}

    rs_match = re.search(r'"repair_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    result["repair_summary"] = (rs_match.group(1) if rs_match else "").replace(
        "\\n", " "
    )

    out_match = re.search(r'"outcome"\s*:\s*"([^"]*)"', raw)
    result["outcome"] = out_match.group(1) if out_match else "unknown"

    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    result["confidence"] = float(conf_match.group(1)) if conf_match else 0.0

    return result if result else None

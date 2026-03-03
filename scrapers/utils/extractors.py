"""
Extraction utilities for automotive diagnostic data from scraped text.

Patterns align with docs/WEB_SCRAPING_PROMPT.md and process_scraped_data.py.
"""
import re
from typing import Any

from scrapers.utils.constants import (
    FAULT_CODE_EXTRACT_PATTERNS,
    OBD_PARAM_PATTERNS,
    VEHICLE_MAKE_PATTERN,
    YEAR_PATTERN,
    ENGINE_PATTERN,
    MILEAGE_PATTERN,
)


def extract_fault_codes(text: str) -> list[str]:
    """
    Extract fault codes from text using standard patterns.

    Args:
        text: Raw text (post body, description, etc.)

    Returns:
        List of unique, normalized fault codes (uppercase).
    """
    if not text or not isinstance(text, str):
        return []
    codes: set[str] = set()
    for pattern in FAULT_CODE_EXTRACT_PATTERNS:
        for m in pattern.finditer(text):
            code = m.group(1).strip().upper()
            codes.add(code)
    return sorted(codes)


def extract_obd_data(text: str) -> dict[str, float]:
    """
    Extract OBD-II parameters from text descriptions.

    Args:
        text: Raw text mentioning OBD readings.

    Returns:
        Dict of parameter name -> float value.
    """
    if not text or not isinstance(text, str):
        return {}
    result: dict[str, float] = {}
    for param, pattern in OBD_PARAM_PATTERNS.items():
        m = pattern.search(text)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                try:
                    val = float(groups[0].replace(",", ""))
                    result[param] = val
                except ValueError:
                    pass
    return result


def extract_vehicle_context(text: str) -> dict[str, Any]:
    """
    Extract vehicle context (make, model, year, engine, mileage) from text.

    Args:
        text: Raw text mentioning vehicle info.

    Returns:
        Dict with make, model, year, engine, mileage (keys present only if found).
    """
    if not text or not isinstance(text, str):
        return {}
    ctx: dict[str, Any] = {}

    make_m = VEHICLE_MAKE_PATTERN.search(text)
    if make_m:
        ctx["make"] = make_m.group(1).strip().title()

    year_m = YEAR_PATTERN.search(text)
    if year_m:
        try:
            ctx["year"] = int(year_m.group(1))
        except ValueError:
            pass

    engine_m = ENGINE_PATTERN.search(text)
    if engine_m:
        ctx["engine"] = engine_m.group(1).strip().upper()

    mileage_m = MILEAGE_PATTERN.search(text)
    if mileage_m:
        groups = [g for g in mileage_m.groups() if g is not None]
        if groups:
            try:
                ctx["mileage"] = int(groups[0].replace(",", ""))
            except ValueError:
                pass

    return ctx


def extract_repair_summary(text: str, max_chars: int = 500) -> str:
    """
    Extract a concise repair summary from longer text.

    Looks for common patterns: "replaced X", "fixed by", "solution was", etc.

    Args:
        text: Raw text (e.g., forum reply, video description).
        max_chars: Maximum length of summary.

    Returns:
        Trimmed summary string, or empty if nothing found.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    # Try to find repair/solution section
    patterns = [
        r'(?:replaced|fixed|repaired|cleaned|reset)\s+[^.]+\.[^.]*\.',
        r'(?:solution|fix|repair)\s*[:\-]\s*[^.\n]+(?:\.[^.\n]+)*',
        r'(?:what\s+i\s+did|steps?\s*[:\-])\s*[^.\n]+(?:\.[^.\n]+)*',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            summary = m.group(0).strip()
            return summary[:max_chars] if len(summary) > max_chars else summary

    # Fallback: first few sentences
    sentences = re.split(r'[.!?]\s+', text)
    summary_parts = []
    total = 0
    for s in sentences:
        if total + len(s) + 1 > max_chars:
            break
        summary_parts.append(s.strip())
        total += len(s) + 1
    return ". ".join(summary_parts) + ("." if summary_parts else "")


def extract_outcome(text: str) -> str:
    """
    Infer outcome (success/failure/partial/unknown) from text.

    Args:
        text: Raw text (e.g., "fixed it", "didn't work").

    Returns:
        One of: success, failure, partial, unknown
    """
    if not text or not isinstance(text, str):
        return "unknown"
    t = text.lower()
    if any(
        x in t
        for x in ("fixed", "resolved", "worked", "solved", "no more", "cleared")
    ):
        return "success"
    if any(
        x in t
        for x in ("didn't work", "still getting", "still have", "failed", "no luck")
    ):
        return "failure"
    if any(
        x in t
        for x in ("partially", "somewhat", "better but", "improved")
    ):
        return "partial"
    return "unknown"

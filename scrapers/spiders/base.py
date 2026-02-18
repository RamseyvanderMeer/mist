"""
Base spider for MIST automotive diagnostic data collection.

Provides shared extraction logic, text cleaning, and error handling.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Any

import scrapy
from scrapy.http import Response

from scrapers.items import MistScrapedItem
from scrapers.utils.extractors import (
    extract_fault_codes,
    extract_obd_data,
    extract_outcome,
    extract_repair_summary,
    extract_vehicle_context,
)

logger = logging.getLogger(__name__)


def clean_text(text: str | None) -> str:
    """Normalize whitespace and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def merge_text(*parts: str | None) -> str:
    """Join non-empty parts with newlines."""
    return "\n".join(p for p in parts if p and clean_text(p))


class MistBaseSpider(scrapy.Spider):
    """
    Base spider for automotive diagnostic sources.

    Subclasses should override parse() and use build_item() to create
    MistScrapedItem instances.
    """

    custom_settings = {
        "DOWNLOAD_DELAY": 2.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
    }

    def build_item(
        self,
        *,
        fault_codes: list[str] | None = None,
        obd_data: dict[str, float] | None = None,
        vehicle_context: dict[str, Any] | None = None,
        repair_summary: str = "",
        outcome: str = "unknown",
        source_url: str = "",
        source_type: str = "forum",
        symptoms: str = "",
        diagnostic_steps: str = "",
        parts_used: list[str] | None = None,
        **kwargs: Any,
    ) -> MistScrapedItem | None:
        """
        Build a MistScrapedItem, optionally enriching from raw text.

        If fault_codes is empty but raw_text is provided, fault codes are extracted.
        Same for obd_data, vehicle_context, repair_summary, outcome.
        """
        item = MistScrapedItem()
        item["fault_codes"] = fault_codes or []
        item["obd_data"] = obd_data or {}
        item["vehicle_context"] = vehicle_context or {}
        item["repair_summary"] = clean_text(repair_summary) or ""
        item["outcome"] = outcome or "unknown"
        item["source_url"] = source_url or ""
        item["source_type"] = source_type or "forum"
        item["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        raw_text = kwargs.get("raw_text", "")
        if raw_text:
            raw_text = merge_text(raw_text) if isinstance(raw_text, (list, tuple)) else str(raw_text)

        if not item["fault_codes"] and raw_text:
            item["fault_codes"] = extract_fault_codes(raw_text)
        if not item["obd_data"] and raw_text:
            item["obd_data"] = extract_obd_data(raw_text)
        if not item["vehicle_context"] and raw_text:
            item["vehicle_context"] = extract_vehicle_context(raw_text)
        if not item["repair_summary"] and raw_text:
            item["repair_summary"] = extract_repair_summary(raw_text)
        if item["outcome"] == "unknown" and raw_text:
            item["outcome"] = extract_outcome(raw_text)

        if symptoms:
            item["symptoms"] = clean_text(symptoms)
        if diagnostic_steps:
            item["diagnostic_steps"] = clean_text(diagnostic_steps)
        if parts_used:
            item["parts_used"] = parts_used

        for k, v in kwargs.items():
            if k != "raw_text" and k in item.fields and k not in item:
                item[k] = v

        return item

    def parse(self, response: Response, **kwargs: Any):
        """Override in subclasses."""
        raise NotImplementedError

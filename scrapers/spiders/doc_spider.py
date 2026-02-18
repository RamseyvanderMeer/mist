"""
Documentation spider for OBD code databases and repair guides.

Scrapes OBD-Codes.com, AutoZone, and similar documentation sites.
Loads fault codes from the ISTA database; falls back to a small hardcoded list
if the database is unavailable.
"""
import logging

import scrapy

from scrapers.spiders.base import MistBaseSpider, merge_text

logger = logging.getLogger(__name__)

# Fallback when ISTA database is unavailable
FALLBACK_P_CODES = [
    "P0300", "P0301", "P0420", "P0430", "P0171", "P0174",
    "P0128", "P0507", "P2187", "P2189", "P2195", "P2197",
    "P2096", "P2098", "P2270", "P2271", "P0170", "P0173",
]


def _load_fault_codes_from_db(code_pattern: str | None = "P%") -> list[str] | None:
    """
    Load fault codes from the ISTA database.
    
    Args:
        code_pattern: SQL LIKE pattern for codes (e.g. "P%" for OBD-II P-codes only).
                     Use None for all codes.
    
    Returns:
        List of fault code strings, or None if the database is unavailable.
    """
    try:
        from src.database.ista_db import IstaDatabase
        db = IstaDatabase()
        if not db.test_connection():
            return None
        codes = db.get_all_fault_codes(code_pattern=code_pattern)
        db.close()
        return codes if codes else None
    except Exception as e:
        logger.warning("Could not load fault codes from database: %s", e)
        return None


class DocSpider(MistBaseSpider):
    """
    Spider for OBD code documentation sites.

    Scrapes fault code definitions, causes, and repair procedures.
    """

    name = "doc"
    handle_httpstatus_list = [404, 500, 502, 503]
    allowed_domains = [
        "obd-codes.com",
        "www.obd-codes.com",
        "engine-codes.com",
        "autozone.com",
    ]

    def start_requests(self):
        """Generate URLs for fault codes from DB, or fallback to hardcoded list."""
        raw = self.settings.get("MIST_DOC_CODE_PATTERN", "P%")
        # "P%" = OBD-II P-codes only; "" or None = all codes
        code_pattern = None if raw in ("", None) else raw
        codes = _load_fault_codes_from_db(code_pattern=code_pattern)
        if not codes:
            codes = FALLBACK_P_CODES
            logger.info(
                "Database unavailable, using %d fallback P-codes",
                len(codes),
            )
        else:
            logger.info(
                "Loaded %d fault codes from database (pattern=%s)",
                len(codes),
                code_pattern or "all",
            )
        
        for code in codes:
            url = f"https://www.obd-codes.com/{code.lower()}"
            yield scrapy.Request(url, callback=self.parse_code_page)

    def parse_code_page(self, response):
        """Parse OBD code documentation page."""
        if response.status >= 400:
            logger.warning("HTTP %d for %s", response.status, response.url)
            return

        title = response.css("h1::text, .page-title::text").get()
        content = response.css(
            ".content, .article-body, main, .code-description, #content"
        )
        if not content:
            content = response.css("body")

        texts = []
        if title:
            texts.append(title)
        for block in content:
            body = block.css("::text").getall()
            if body:
                texts.append(" ".join(body))

        raw_text = merge_text(*texts)
        if not raw_text:
            return

        item = self.build_item(
            raw_text=raw_text,
            source_url=response.url,
            source_type="documentation",
        )
        if item and item.get("fault_codes"):
            yield item

"""
Integration tests for DocSpider parsing.

Tests spider parsing logic with sample HTML - no network calls.
"""
import pytest
from pathlib import Path

from scrapy.http import TextResponse

from scrapers.spiders.doc_spider import DocSpider


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_doc_spider_parses_p0300_page():
    """DocSpider yields MistScrapedItem with fault codes from sample HTML."""
    html_path = FIXTURES_DIR / "p0300_sample.html"
    html = html_path.read_text(encoding="utf-8")
    url = "https://www.obd-codes.com/p0300"

    spider = DocSpider()
    response = TextResponse(url=url, body=html.encode("utf-8"), encoding="utf-8")

    items = list(spider.parse_code_page(response))

    assert len(items) == 1
    item = items[0]
    assert "fault_codes" in item
    assert "P0300" in item["fault_codes"]
    assert item["source_url"] == url
    assert item["source_type"] == "documentation"
    assert item["repair_summary"]
    assert "timestamp" in item


def test_doc_spider_skips_404():
    """DocSpider yields nothing for 404 response."""
    html = "<html><body>Not Found</body></html>"
    url = "https://www.obd-codes.com/p9999"

    spider = DocSpider()
    response = TextResponse(
        url=url,
        body=html.encode("utf-8"),
        encoding="utf-8",
        status=404,
    )

    items = list(spider.parse_code_page(response))

    assert len(items) == 0


def test_doc_spider_skips_500():
    """DocSpider yields nothing for 500 response."""
    html = "<html><body>Server Error</body></html>"
    url = "https://www.obd-codes.com/p0300"

    spider = DocSpider()
    response = TextResponse(
        url=url,
        body=html.encode("utf-8"),
        encoding="utf-8",
        status=500,
    )

    items = list(spider.parse_code_page(response))

    assert len(items) == 0


def test_doc_spider_fallback_to_body():
    """DocSpider uses body when .content is missing."""
    html = """
    <html>
    <body>
    <h1>P0420 - Catalyst System Efficiency</h1>
    <p>P0420 means the catalytic converter is not working properly.</p>
    </body>
    </html>
    """
    url = "https://www.obd-codes.com/p0420"

    spider = DocSpider()
    response = TextResponse(url=url, body=html.encode("utf-8"), encoding="utf-8")

    items = list(spider.parse_code_page(response))

    assert len(items) == 1
    assert "P0420" in items[0]["fault_codes"]

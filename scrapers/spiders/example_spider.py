"""
Example Scrapy spider - placeholder for automotive sources.
"""
import scrapy


class ExampleSpider(scrapy.Spider):
    """Minimal spider for Cloud Run Job smoke test."""

    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com/"]

    def parse(self, response):
        yield {
            "url": response.url,
            "html": response.text[:5000],
            "raw_text": response.css("body::text").getall(),
        }

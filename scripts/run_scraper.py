#!/usr/bin/env python3
"""
Entry point for MIST scraper - runs Scrapy crawl, optionally uploads to GCS.

Supports running specific spiders (forum, doc, example) or all.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for DATABASE_URL, GEMINI_API_KEY, etc.
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

OUTPUT_FILE = "/tmp/scraper_output.jsonl"


def upload_to_gcs(local_path: str, bucket_name: str) -> None:
    """Upload output file to GCS bucket."""
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("output.jsonl")
        blob.upload_from_filename(local_path)
    except Exception as e:
        print(f"GCS upload failed: {e}", file=sys.stderr)


def get_spider_class(name: str):
    """Resolve spider name to class."""
    spiders = {
        "forum": ("scrapers.spiders.forum_spider", "ForumSpider"),
        "doc": ("scrapers.spiders.doc_spider", "DocSpider"),
        "example": ("scrapers.spiders.example_spider", "ExampleSpider"),
    }
    if name not in spiders:
        raise ValueError(f"Unknown spider: {name}. Choose from: {list(spiders)}")
    mod_name, cls_name = spiders[name]
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run MIST web scraper for automotive diagnostic data"
    )
    parser.add_argument(
        "--spider",
        choices=["forum", "doc", "example"],
        default="forum",
        help="Spider to run (default: forum)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/raw_data"),
        help="Output directory for raw data (default: data/training/raw_data)",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=None,
        help="Stop after N items (sets CLOSESPIDER_ITEMCOUNT)",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Stop after N pages (sets CLOSESPIDER_PAGECOUNT)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Specific URL to scrape (overrides default start_urls, forum spider only)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Use search-based discovery (forum spider: fault-code search URLs)",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Use targeted subforums (engine/diagnostics) instead of general",
    )
    parser.add_argument(
        "--search-codes",
        action="store_true",
        help="Search each fault code (P0300, 2A87, etc.) on every forum that supports search",
    )
    parser.add_argument(
        "--re-scrape",
        action="store_true",
        help="Ignore previously scraped URLs (re-process them)",
    )
    args = parser.parse_args()

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    settings = get_project_settings()
    settings.set("MIST_RAW_DATA_DIR", str(args.output_dir))

    if args.limit_items is not None:
        settings.set("CLOSESPIDER_ITEMCOUNT", args.limit_items)
    if args.limit_pages is not None:
        settings.set("CLOSESPIDER_PAGECOUNT", args.limit_pages)

    # Example spider uses FEEDS for Cloud Run; others use JsonlWriterPipeline
    if args.spider == "example":
        settings.set("FEEDS", {OUTPUT_FILE: {"format": "jsonlines"}})

    try:
        process = CrawlerProcess(settings)
        spider_cls = get_spider_class(args.spider)
        crawl_kwargs = {}
        if args.spider == "forum":
            if args.url:
                crawl_kwargs["start_url"] = args.url
            else:
                crawl_kwargs["use_search"] = args.search
                crawl_kwargs["use_targeted"] = args.targeted
                crawl_kwargs["use_search_codes"] = args.search_codes
                crawl_kwargs["re_scrape"] = args.re_scrape
            crawl_kwargs["output_dir"] = str(args.output_dir)
        process.crawl(spider_cls, **crawl_kwargs)
        process.start()
    except Exception as e:
        print(f"Scraper failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.spider == "example":
        bucket = os.environ.get("BUCKET", "").replace("gs://", "").split("/")[0]
        if bucket and Path(OUTPUT_FILE).exists():
            upload_to_gcs(OUTPUT_FILE, bucket)

    sys.exit(0)


if __name__ == "__main__":
    main()

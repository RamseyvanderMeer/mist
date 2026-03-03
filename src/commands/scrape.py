"""Run MIST web scraper (Scrapy spiders)."""
import os
import sys
from pathlib import Path


def _get_spider_class(name: str):
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


def _upload_to_gcs(local_path: str, bucket_name: str) -> None:
    """Upload output file to GCS bucket."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("output.jsonl")
        blob.upload_from_filename(local_path)
    except Exception as e:
        print(f"GCS upload failed: {e}", file=sys.stderr)


def run(
    spider: str = "forum",
    output_dir: Path | str = "data/training/raw_data",
    limit_items: int | None = None,
    limit_pages: int | None = None,
    url: str | None = None,
    search: bool = False,
    targeted: bool = False,
    search_codes: bool = False,
    re_scrape: bool = False,
) -> int:
    """Run scraper. Returns 0 on success."""
    from dotenv import load_dotenv
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    output_dir = Path(output_dir)
    settings = get_project_settings()
    settings.set("MIST_RAW_DATA_DIR", str(output_dir))

    if limit_items is not None:
        settings.set("CLOSESPIDER_ITEMCOUNT", limit_items)
    if limit_pages is not None:
        settings.set("CLOSESPIDER_PAGECOUNT", limit_pages)

    output_file = "/tmp/scraper_output.jsonl"
    if spider == "example":
        settings.set("FEEDS", {output_file: {"format": "jsonlines"}})

    try:
        process = CrawlerProcess(settings)
        spider_cls = _get_spider_class(spider)
        crawl_kwargs: dict = {}
        if spider == "forum":
            if url:
                crawl_kwargs["start_url"] = url
            else:
                crawl_kwargs["use_search"] = search
                crawl_kwargs["use_targeted"] = targeted
                crawl_kwargs["use_search_codes"] = search_codes
                crawl_kwargs["re_scrape"] = re_scrape
            crawl_kwargs["output_dir"] = str(output_dir)
        process.crawl(spider_cls, **crawl_kwargs)
        process.start()
    except Exception as e:
        print(f"Scraper failed: {e}", file=sys.stderr)
        return 1

    if spider == "example":
        bucket = os.environ.get("BUCKET", "").replace("gs://", "").split("/")[0]
        if bucket and Path(output_file).exists():
            _upload_to_gcs(output_file, bucket)

    return 0

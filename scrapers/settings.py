"""
Scrapy settings for MIST scraper.
"""
import os

# Load .env so DATABASE_URL is available for Postgres pipeline
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")

BOT_NAME = "mist_scraper"
SPIDER_MODULES = ["scrapers.spiders"]
NEWSPIDER_MODULE = "scrapers.spiders"
ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 8
DOWNLOAD_DELAY = 2.0

# Storage: "postgres" when DATABASE_URL set, else "jsonl"
_MIST_STORAGE = "postgres" if os.environ.get("DATABASE_URL", "").startswith("postgresql") else "jsonl"

ITEM_PIPELINES = {
    "scrapers.pipelines.langgraph_pipeline.LLMExtractionPipeline": 300,
    "scrapers.pipelines.validation.MistValidationPipeline": 400,
    "scrapers.pipelines.io.JsonlWriterPipeline": 500,
    "scrapers.pipelines.postgres.PostgresWriterPipeline": 510,
}

# Disable JSONL when using Postgres
if _MIST_STORAGE == "postgres":
    ITEM_PIPELINES.pop("scrapers.pipelines.io.JsonlWriterPipeline", None)

DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "scrapy.downloadermiddlewares.retry.RetryMiddleware": 550,
}
USER_AGENT = "MIST-Scraper/1.0 (+https://github.com/mist-diagnostics; research)"

# MIST-specific settings
CONCURRENT_REQUESTS_PER_DOMAIN = 2  # Limit per-domain concurrency to avoid bans
RANDOMIZE_DOWNLOAD_DELAY = True     # Randomize delay (0.5 * DOWNLOAD_DELAY to 1.5 * DOWNLOAD_DELAY)
MIST_MIN_QUALITY = 0.6
MIST_RAW_DATA_DIR = "data/training/raw_data"
# Doc spider: SQL LIKE pattern for fault codes from DB. "P%" = OBD-II P-codes only; "" = all codes
MIST_DOC_CODE_PATTERN = "P%"

# LLM extraction (Gemini)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_EXTRACTION_ENABLED = True
LLM_MIN_CONFIDENCE = 0.8
LLM_PREFILTER_ENABLED = True  # Skip LLM when no fault codes/repair-like text (saves tokens)

# Postgres (for PostgresWriterPipeline)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Proxy (Smart fallback; SCRAPERAPI_KEY_ONE and SCRAPERAPI_KEY_TWO used after 3 failures with primary)
SCRAPERAPI_KEY = os.environ.get("SCRAPERAPI_KEY")
SCRAPERAPI_KEY_ONE = os.environ.get("SCRAPERAPI_KEY_ONE")
SCRAPERAPI_KEY_TWO = os.environ.get("SCRAPERAPI_KEY_TWO")
if SCRAPERAPI_KEY:
    # Use SmartProxyMiddleware for selective proxy usage on 403/429
    DOWNLOADER_MIDDLEWARES["scrapers.middlewares.SmartProxyMiddleware"] = 610
    # Note: We keep standard concurrency/delay settings to be polite for direct requests.
    # The middleware handles retrying with proxy only when needed.


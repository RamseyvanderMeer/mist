"""
JSONL output pipeline for MIST scraped items.

Writes validated items to source-specific JSONL files in data/training/raw_data/.
"""
import json
import logging
from pathlib import Path

from scrapy import Spider

logger = logging.getLogger(__name__)


class JsonlWriterPipeline:
    """Append validated MistScrapedItem to JSONL files by source type."""

    def __init__(self, output_dir: str | Path = "data/training/raw_data"):
        self.output_dir = Path(output_dir)

    @classmethod
    def from_crawler(cls, crawler):
        output = crawler.settings.get(
            "MIST_RAW_DATA_DIR",
            crawler.settings.get("DATA_DIR", "data/training/raw_data"),
        )
        return cls(output_dir=output)

    def process_item(self, item, spider: Spider):
        # Accept fault_code records or cause_to_solution records (may have empty fault_codes)
        record_type = item.get("record_type", "fault_code")
        if record_type not in ("fault_code", "cause_to_solution"):
            return item
        if "fault_codes" not in item:
            return item

        source_type = (item.get("source_type") or "forum").lower()
        if source_type not in ("forum", "documentation", "video", "tsb"):
            source_type = "forum"

        subdirs = {
            "forum": "forums",
            "documentation": "documentation",
            "video": "videos",
            "tsb": "tsbs",
        }
        subdir = self.output_dir / subdirs[source_type]

        subdir.mkdir(parents=True, exist_ok=True)
        filename = f"{source_type}_{spider.name}.jsonl"
        filepath = subdir / filename

        record = dict(item)
        # Remove raw_text to avoid bloating output (used only for LLM pipeline)
        record.pop("raw_text", None)
        # Add repair_guide for process_scraped_data.py compatibility
        if "repair_guide" not in record and record.get("repair_summary"):
            record["repair_guide"] = record["repair_summary"]

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (IOError, OSError) as e:
            logger.error("Failed to write item to %s: %s", filepath, e)
            raise

        logger.debug("Wrote item to %s", filepath)
        return item

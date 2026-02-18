"""
PostgreSQL output pipeline for MIST scraped items.

Writes validated items to scraped_records table when DATABASE_URL is set.
"""
import json
import logging
import os
from datetime import datetime, timezone

from scrapy import Spider

logger = logging.getLogger(__name__)


def _serialize_json(obj):
    """Serialize dict/list to JSON string for JSONB."""
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def _parse_ts(ts):
    """Parse timestamp to ISO string for Postgres."""
    if ts is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


class PostgresWriterPipeline:
    """Write validated MistScrapedItem to PostgreSQL scraped_records table."""

    def __init__(self, database_url: str | None = None, log_every: int = 10):
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        self._enabled = bool(self._database_url and self._database_url.startswith("postgresql"))
        self._log_every = log_every
        self._items_written = 0

    @classmethod
    def from_crawler(cls, crawler):
        url = crawler.settings.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
        every = crawler.settings.getint("MIST_PROGRESS_LOG_EVERY", 10)
        return cls(database_url=url, log_every=every)

    def open_spider(self, spider: Spider):
        self._items_written = 0
        if not self._enabled:
            logger.info("PostgresWriterPipeline disabled (no DATABASE_URL)")
            return
        self._ensure_table()

    def _ensure_table(self):
        """Create table if not exists (idempotent)."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self._database_url)
            migration_path = (
                __file__.replace("postgres.py", "").replace("pipelines", "")
                + "../../../scripts/migrations/create_scraped_records_postgres.sql"
            )
            # Resolve path from project root
            from pathlib import Path

            root = Path(__file__).parent.parent.parent
            migration_file = root / "scripts" / "migrations" / "create_scraped_records_postgres.sql"
            if migration_file.exists():
                with open(migration_file, "r") as f:
                    sql = f.read()
                with engine.connect() as conn:
                    conn.execute(text(sql))
                    conn.commit()
                logger.info("Postgres scraped_records table ready")
        except Exception as e:
            logger.warning("Could not ensure scraped_records table: %s", e)

    def process_item(self, item, spider: Spider):
        if not self._enabled:
            return item

        record_type = item.get("record_type", "fault_code")
        if record_type not in ("fault_code", "cause_to_solution"):
            return item
        if "fault_codes" not in item:
            return item

        source_url = (item.get("source_url") or "").strip()
        if not source_url:
            logger.warning("Skipping item with empty source_url")
            return item

        fault_codes = item.get("fault_codes") or []
        obd_data = item.get("obd_data") or {}
        vehicle_context = item.get("vehicle_context") or {}
        repair_summary = (item.get("repair_summary") or "").strip()
        repair_guide = item.get("repair_guide") or repair_summary
        if isinstance(repair_guide, dict):
            repair_guide = repair_guide.get("title", "") or repair_guide.get("description", "") or json.dumps(repair_guide)

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self._database_url)
            fc_json = _serialize_json(fault_codes)
            obd_json = _serialize_json(obd_data)
            vc_json = _serialize_json(vehicle_context)

            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO scraped_records (
                            source_url, fault_codes, obd_data, vehicle_context,
                            repair_summary, outcome, source_type, record_type,
                            symptoms, confidence_score, heuristic_score, llm_confidence,
                            repair_guide, timestamp
                        ) VALUES (
                            :source_url, CAST(:fault_codes AS jsonb), CAST(:obd_data AS jsonb),
                            CAST(:vehicle_context AS jsonb), :repair_summary, :outcome,
                            :source_type, :record_type, :symptoms, :confidence_score,
                            :heuristic_score, :llm_confidence, :repair_guide,
                            CAST(:timestamp AS timestamptz)
                        )
                        ON CONFLICT (source_url) DO UPDATE SET
                            fault_codes = EXCLUDED.fault_codes,
                            obd_data = EXCLUDED.obd_data,
                            vehicle_context = EXCLUDED.vehicle_context,
                            repair_summary = EXCLUDED.repair_summary,
                            outcome = EXCLUDED.outcome,
                            record_type = EXCLUDED.record_type,
                            symptoms = EXCLUDED.symptoms,
                            confidence_score = EXCLUDED.confidence_score,
                            heuristic_score = EXCLUDED.heuristic_score,
                            llm_confidence = EXCLUDED.llm_confidence,
                            repair_guide = EXCLUDED.repair_guide,
                            timestamp = EXCLUDED.timestamp
                    """),
                    {
                        "source_url": source_url,
                        "fault_codes": fc_json,
                        "obd_data": obd_json,
                        "vehicle_context": vc_json,
                        "repair_summary": repair_summary or None,
                        "outcome": (item.get("outcome") or "unknown") or None,
                        "source_type": (item.get("source_type") or "forum") or "forum",
                        "record_type": record_type,
                        "symptoms": (item.get("symptoms") or "").strip() or None,
                        "confidence_score": item.get("confidence_score"),
                        "heuristic_score": item.get("heuristic_score"),
                        "llm_confidence": item.get("llm_confidence"),
                        "repair_guide": repair_guide or None,
                        "timestamp": _parse_ts(item.get("timestamp")),
                    },
                )
                conn.commit()
            self._items_written += 1
            if self._log_every and self._items_written % self._log_every == 0:
                logger.info("Progress: %d items written to Postgres", self._items_written)
            else:
                logger.debug("Wrote item to Postgres: %s", source_url[:60])
        except Exception as e:
            logger.error("Failed to write item to Postgres: %s", e)
            raise

        return item

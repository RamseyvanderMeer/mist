#!/usr/bin/env python3
"""
Export scraped_records from Postgres to JSONL for process_scraped_data.py.

Usage:
    python scripts/export_scraped_from_postgres.py
    python scripts/export_scraped_from_postgres.py -o data/training/raw_data/forums/forum_from_db.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    print("DATABASE_URL not set. Add it to .env", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export scraped_records from Postgres to JSONL")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=ROOT / "data" / "training" / "raw_data" / "forums" / "forum_from_db.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        help="Filter by source_type (forum, documentation, etc.)",
    )
    args = parser.parse_args()

    from sqlalchemy import create_engine, text

    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM scraped_records"
    params = {}
    if args.source_type:
        query += " WHERE source_type = :source_type"
        params["source_type"] = args.source_type
    query += " ORDER BY created_at DESC"

    records = []
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        columns = result.keys()
        for row in result:
            rec = dict(zip(columns, row))
            # Convert to JSONL-compatible format (Postgres JSONB returns dict/list)
            def _json_val(val, default):
                if val is None:
                    return default
                if isinstance(val, (list, dict)):
                    return val
                return json.loads(val) if isinstance(val, str) else default

            out = {
                "source_url": rec.get("source_url"),
                "fault_codes": _json_val(rec.get("fault_codes"), []),
                "obd_data": _json_val(rec.get("obd_data"), {}),
                "vehicle_context": _json_val(rec.get("vehicle_context"), {}),
                "repair_summary": rec.get("repair_summary"),
                "repair_guide": rec.get("repair_guide") or rec.get("repair_summary"),
                "outcome": rec.get("outcome"),
                "source_type": rec.get("source_type"),
                "record_type": rec.get("record_type"),
                "symptoms": rec.get("symptoms"),
                "timestamp": str(rec.get("timestamp")) if rec.get("timestamp") else None,
            }
            records.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

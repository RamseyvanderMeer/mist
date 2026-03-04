#!/usr/bin/env python3
"""Run Postgres migration for retrieved_records table.

Requires DATABASE_URL in environment (e.g. from .env).

    python scripts/run_retrieved_records_migration.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    print("DATABASE_URL not set or not PostgreSQL. Set it in .env", file=sys.stderr)
    sys.exit(1)

from sqlalchemy import create_engine, text

migration_file = ROOT / "scripts" / "migrations" / "create_retrieved_records_postgres.sql"
with open(migration_file, "r") as f:
    sql = f.read()

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    for stmt in (s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")):
        try:
            conn.execute(text(stmt))
            conn.commit()
            print("OK:", stmt[:60].replace("\n", " ") + "...")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("Exists (skip):", stmt[:60].replace("\n", " "))
            else:
                raise

print("retrieved_records migration complete.")

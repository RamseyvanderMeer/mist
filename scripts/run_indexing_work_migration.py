#!/usr/bin/env python3
"""
Run Postgres migration for indexing_work table.

Creates the table used for multi-machine indexing coordination.
Requires DATABASE_URL in environment (e.g. from .env).

Usage:
    python scripts/run_indexing_work_migration.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
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

migration_file = ROOT / "scripts" / "migrations" / "create_indexing_work_postgres.sql"
if not migration_file.exists():
    print(f"Migration file not found: {migration_file}", file=sys.stderr)
    sys.exit(1)

from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    with open(migration_file, "r") as f:
        sql = f.read()
    for stmt in (s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")):
        try:
            conn.execute(text(stmt))
            conn.commit()
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

print("Migration completed: indexing_work table ready.")

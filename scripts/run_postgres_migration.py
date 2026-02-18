#!/usr/bin/env python3
"""
Run Postgres migration for scraped_records table.

Requires DATABASE_URL in environment (e.g. from .env).
Usage:
    python scripts/run_postgres_migration.py
    # or with explicit URL:
    DATABASE_URL=postgresql://... python scripts/run_postgres_migration.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    print("DATABASE_URL not set or not PostgreSQL. Set it in .env or environment.", file=sys.stderr)
    sys.exit(1)

create_file = ROOT / "scripts" / "migrations" / "create_scraped_records_postgres.sql"
alter_file = ROOT / "scripts" / "migrations" / "alter_scraped_records_add_columns.sql"
if not create_file.exists():
    print(f"Migration file not found: {create_file}", file=sys.stderr)
    sys.exit(1)

from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # 1. Create table if not exists
    with open(create_file, "r") as f:
        sql = f.read()
    stmts = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
    # Only run CREATE TABLE (first real statement)
    create_table = next((s for s in stmts if "CREATE TABLE" in s), None)
    if create_table:
        try:
            conn.execute(text(create_table))
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
    conn.commit()

    # 2. Add missing columns (must run before CREATE INDEX on new columns)
    if alter_file.exists():
        with open(alter_file, "r") as f:
            alter_sql = f.read()
        for stmt in (s.strip() for s in alter_sql.split(";") if s.strip() and not s.strip().startswith("--")):
            try:
                conn.execute(text(stmt))
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
        conn.commit()

    # 3. Create indexes (may already exist)
    for stmt in stmts:
        if "CREATE INDEX" in stmt:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
    conn.commit()

print("Migration completed: scraped_records table ready.")

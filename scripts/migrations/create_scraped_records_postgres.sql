-- scraped_records table for PostgreSQL (NeonDB)
-- Stores scraped automotive diagnostic data from forums, documentation, etc.
-- Run with: psql $DATABASE_URL -f create_scraped_records_postgres.sql

-- Drop and recreate if schema changed (optional - comment out to preserve data)
-- DROP TABLE IF EXISTS scraped_records;

CREATE TABLE IF NOT EXISTS scraped_records (
    id SERIAL PRIMARY KEY,
    source_url TEXT NOT NULL UNIQUE,
    fault_codes JSONB DEFAULT '[]',
    obd_data JSONB DEFAULT '{}',
    vehicle_context JSONB DEFAULT '{}',
    repair_summary TEXT,
    outcome TEXT,
    source_type TEXT DEFAULT 'forum',
    record_type TEXT DEFAULT 'fault_code',
    symptoms TEXT,
    confidence_score REAL,
    heuristic_score REAL,
    llm_confidence REAL,
    repair_guide TEXT,
    matched_guide_id TEXT,
    matched_guide_title TEXT,
    match_reasoning JSONB,
    timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scraped_records_source_url ON scraped_records(source_url);
CREATE INDEX IF NOT EXISTS idx_scraped_records_source_type ON scraped_records(source_type);
CREATE INDEX IF NOT EXISTS idx_scraped_records_record_type ON scraped_records(record_type);
CREATE INDEX IF NOT EXISTS idx_scraped_records_created_at ON scraped_records(created_at);

-- GIN index for fault_codes JSONB queries (e.g. WHERE fault_codes ? 'P0300')
CREATE INDEX IF NOT EXISTS idx_scraped_records_fault_codes ON scraped_records USING GIN (fault_codes);

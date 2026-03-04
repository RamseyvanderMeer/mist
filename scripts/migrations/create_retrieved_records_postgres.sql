-- retrieved_records table for PostgreSQL (NeonDB)
-- Stores retrieval evaluation results: ground truth from scraped_records vs. retrieved guides.
-- Run with: psql $DATABASE_URL -f create_retrieved_records_postgres.sql

CREATE TABLE IF NOT EXISTS retrieved_records (
    id SERIAL PRIMARY KEY,
    scraped_record_id INTEGER,
    source_url TEXT,
    fault_codes JSONB DEFAULT '[]',
    description TEXT,
    expected_guide_id TEXT NOT NULL,
    expected_guide_title TEXT,
    retrieved_guide_ids JSONB DEFAULT '[]',
    retrieved_scores JSONB DEFAULT '[]',
    hit_at_1 BOOLEAN,
    hit_at_5 BOOLEAN,
    hit_at_10 BOOLEAN,
    reciprocal_rank REAL,
    run_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrieved_records_run_id ON retrieved_records(run_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_records_expected_guide ON retrieved_records(expected_guide_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_records_created_at ON retrieved_records(created_at);

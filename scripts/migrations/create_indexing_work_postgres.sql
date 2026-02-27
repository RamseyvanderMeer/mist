-- indexing_work table for PostgreSQL
-- Coordinates multi-machine repair guide indexing. Workers claim rows via
-- SELECT ... FOR UPDATE SKIP LOCKED to avoid overlap.
-- Run with: psql $DATABASE_URL -f create_indexing_work_postgres.sql

CREATE TABLE IF NOT EXISTS indexing_work (
    procedure_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    worker_id TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_indexing_work_status ON indexing_work(status);
CREATE INDEX IF NOT EXISTS idx_indexing_work_worker ON indexing_work(worker_id);

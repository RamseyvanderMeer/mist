-- Add missing columns to scraped_records (for existing tables)
-- Safe to run multiple times (IF NOT EXISTS)

ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS record_type TEXT DEFAULT 'fault_code';
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS symptoms TEXT;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS confidence_score REAL;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS heuristic_score REAL;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS llm_confidence REAL;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS repair_guide TEXT;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS matched_guide_id TEXT;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS matched_guide_title TEXT;
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS match_reasoning JSONB;

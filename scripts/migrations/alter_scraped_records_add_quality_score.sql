-- Add quality_score column to scraped_records for processed/validated records.
ALTER TABLE scraped_records ADD COLUMN IF NOT EXISTS quality_score REAL;

-- ranking_weights table for PostgreSQL (NeonDB)
-- Stores tunable ranking weights for repair guide lookup.
-- DB weights override config/retrieval_config.yaml when DATABASE_URL is set.

CREATE TABLE IF NOT EXISTS ranking_weights (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    embedding_similarity REAL NOT NULL DEFAULT 0.4,
    rerank_score REAL NOT NULL DEFAULT 0.3,
    kg_path_score REAL NOT NULL DEFAULT 0.2,
    feedback_score REAL NOT NULL DEFAULT 0.1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed default weights (ON CONFLICT requires unique constraint on name)
INSERT INTO ranking_weights (name, embedding_similarity, rerank_score, kg_path_score, feedback_score, is_active)
VALUES ('default', 0.4, 0.3, 0.2, 0.1, true)
ON CONFLICT (name) DO NOTHING;

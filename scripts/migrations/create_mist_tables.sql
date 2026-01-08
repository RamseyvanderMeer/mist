-- MIST Database Schema Extensions
-- This migration creates the tables needed for the MIST system
-- Can be run idempotently (safe to run multiple times)

-- Table: feedback_sessions
-- Stores conversational RAG sessions with fault codes, OBD data, clarification questions, and user responses
CREATE TABLE IF NOT EXISTS feedback_sessions (
    session_id TEXT PRIMARY KEY,
    fault_codes TEXT,  -- JSON array
    obd_data TEXT,     -- JSON object
    clarification_questions TEXT,  -- JSON array
    user_responses TEXT,  -- JSON array
    recommended_guides TEXT,  -- JSON array
    selected_guide TEXT,
    explicit_rating INTEGER,  -- 1-5
    repair_outcome TEXT,  -- success/failure/partial
    conversation_corrections TEXT,  -- JSON array
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Table: mist_embeddings
-- Stores procedure embeddings with versioning support
-- Note: procedure_id references XEP_INFOOBJECTS(ID) logically (no FK constraint due to cross-database)
CREATE TABLE IF NOT EXISTS mist_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id TEXT NOT NULL,
    embedding BLOB,  -- 768-dim vector (numpy array)
    embedding_version INTEGER,  -- Version for fine-tuning
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Table: mist_feedback
-- Stores individual feedback entries linked to sessions
CREATE TABLE IF NOT EXISTS mist_feedback (
    feedback_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    procedure_id TEXT,  -- Logical reference to XEP_INFOOBJECTS(ID)
    rating INTEGER,  -- 1-5
    repair_outcome TEXT,  -- success/failure/partial
    feedback_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
);

-- Table: mist_training_checkpoints
-- Tracks embedding model training checkpoints
CREATE TABLE IF NOT EXISTS mist_training_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    epoch INTEGER,
    loss REAL,
    validation_loss REAL,
    embedding_version INTEGER,
    checkpoint_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance

-- Indexes for mist_embeddings
CREATE INDEX IF NOT EXISTS idx_mist_embeddings_procedure ON mist_embeddings(procedure_id);
CREATE INDEX IF NOT EXISTS idx_mist_embeddings_version ON mist_embeddings(embedding_version);

-- Indexes for mist_feedback
CREATE INDEX IF NOT EXISTS idx_mist_feedback_session ON mist_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_mist_feedback_procedure ON mist_feedback(procedure_id);

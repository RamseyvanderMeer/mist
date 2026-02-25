# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MIST (Multi-modal Intelligent Service Technician) is a Python 3.12 FastAPI application for AI-powered automotive diagnostics. See `README.md` for full details.

### Virtual environment

Always activate the venv before running any Python commands:

```bash
source .venv/bin/activate
```

### Running the application

```bash
PYTHONPATH=/workspace uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

The `PYTHONPATH=/workspace` prefix is required for module imports to resolve correctly.

### Running tests

```bash
PYTHONPATH=/workspace python -m pytest tests/ --ignore=tests/e2e -q
```

- The `tests/e2e/` tests run web scrapers with network access and will timeout in cloud environments; always skip them.
- There are ~10 pre-existing test failures in `test_configs.py`, `test_session_manager.py`, `test_retrieval.py`, and `test_prompt_templates.py` — these are bugs in existing code, not environment issues.
- Model-dependent tests (e.g., `test_embeddings.py`, `test_embedding_trainer.py`, `test_reranker.py`) may take a long time as they download HuggingFace models on first run.

### Running lint

No linting config is checked in. `ruff` is installed in the venv:

```bash
ruff check src/ --select E,F
```

### Database migrations

```bash
PYTHONPATH=/workspace python scripts/run_migrations.py
```

This creates/updates `data/databases/mist_data.db` (SQLite). Migrations are idempotent.

### Key gotchas

- The BMW ISTA diagnostic database (`data/databases/DiagDocDb_Decrypted.sqlite`) is a multi-GB file excluded from git. The `/query` endpoint depends on it for full diagnostic pipeline functionality. Feedback endpoints (`/feedback/*`) and `/health` work without it.
- No `.env` file is committed; LLM API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are only required for the `/query` and `/clarify` endpoints (LLM-based clarification). The feedback system and health check work without any API keys.
- The project uses local file-based Qdrant (`data/vector_store/`) by default. No external Qdrant server is needed for development.

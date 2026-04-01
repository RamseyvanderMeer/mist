# AGENTS.md

## Cursor Cloud specific instructions

### Documentation index

For full project documentation, see [docs/agent.md](docs/agent.md). It indexes all .md files by topic (architecture, databases, scraping, training, etc.) for quick lookup.

For **path-grounded architecture and flows** (optimized for AI onboarding), read [docs/SPEC.md](docs/SPEC.md) and [docs/REPO_RULES_FOR_AI.md](docs/REPO_RULES_FOR_AI.md). Cursor loads `.cursor/rules/` including pre/post context gates that reference these files.

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

### Git hooks (Lefthook)

After installing dev tools, register hooks once per clone:

```bash
pip install -e ".[dev]"
lefthook install
```

`pre-commit` runs **Ruff** on **staged** `*.py` files (see `lefthook.yml` and `[tool.ruff]` in `pyproject.toml`). To skip hooks for a single commit: `LEFTHOOK=0 git commit ...`. To run the same checks manually: `lefthook run pre-commit`.

### Running lint

```bash
ruff check src/ tests/
ruff format --check src/ tests/   # optional; many files may still need formatting
```

Rules are defined in `pyproject.toml` (`E` + `F`, with `E501` line-length ignored for now).

### CLI commands

Use `mist-cli` (avoids conflict with npm `mist`). After `pip install -e .`:
  - `mist-cli fetch-bmwfault --limit 10`
  - `mist-cli migrate`, `mist-cli index`, etc.
  - Or: `python mist.py fetch-bmwfault` from project root

### Database migrations

```bash
mist-cli migrate
# Or: PYTHONPATH=/workspace python scripts/run_migrations.py
```

This creates/updates `data/databases/mist_data.db` (SQLite). Migrations are idempotent.

### Key gotchas

- The BMW ISTA diagnostic database (`data/databases/DiagDocDb_Decrypted.sqlite`) is a multi-GB file excluded from git. The `/query` endpoint depends on it for full diagnostic pipeline functionality. Feedback endpoints (`/feedback/*`) and `/health` work without it.
- No `.env` file is committed; LLM API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are only required for the `/query` and `/clarify` endpoints (LLM-based clarification). The feedback system and health check work without any API keys.
- The project uses ChromaDB Cloud for the vector store. Set `CHROMA_DB_API_KEY` and `CHROMA_DB_TENANT` in `.env`.

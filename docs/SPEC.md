# MIST — Repository specification (for humans & AI)

Concise, path-grounded description of the codebase. For narrative architecture, see [ARCHITECTURE.md](ARCHITECTURE.md). For doc index, see [agent.md](agent.md). For behavioral rules in this repo, see [REPO_RULES_FOR_AI.md](REPO_RULES_FOR_AI.md).

---

## 1. Overview

**MIST** (Multi-modal Intelligent Service Technician) is a **Python 3.12** service that recommends BMW repair guides from **fault codes**, **OBD-style sensor data**, and **natural-language symptoms**. It uses **embedding + vector search (ChromaDB)**, **re-ranking**, **knowledge-graph signals** from the BMW ISTA diagnostic database, and **LLM-backed** clarification and prompts.

**Shape:** Single **FastAPI** application (`src/api/server.py`) plus **Typer CLI** (`mist-cli` → `src/cli/main.py`), **batch scripts** under `scripts/`, optional **Scrapy** scrapers under `scrapers/`, and **Terraform** for cloud deployment. Not a monorepo of multiple apps—one product with clear Python packages.

---

## 2. Tech stack

| Layer | Technology | Anchors in repo |
|-------|------------|-----------------|
| Runtime | Python ≥3.12 | `pyproject.toml` (`requires-python`) |
| Lint / git hooks | Ruff + Lefthook (optional dev extra) | `pyproject.toml` (`[tool.ruff]`, `dev` extra), `lefthook.yml` |
| Web API | FastAPI, Uvicorn | `src/api/server.py` |
| CLI | Typer | `src/cli/main.py`, `src/commands/` |
| Config | YAML | `config/*.yaml`, `src/paths.py` (`Paths`) |
| Vector DB | ChromaDB (typically Cloud) | `src/retrieval/chroma_store.py`, `config/retrieval_config.yaml` |
| Embeddings | Multiple encoders (e.g. Qwen3, HF, fault/OBD/multimodal) | `src/embeddings/` |
| BMW diagnostics | Large local SQLite (ISTA) | `src/database/ista_db.py`, `docs/DATABASE.md` |
| Feedback / sessions (app) | SQLite + SQLAlchemy | `src/feedback/collector.py`, `src/database/schema.py`, `src/database/connection.py` |
| Users / auth (API) | PostgreSQL + SQLAlchemy models | `src/database/pg_connection.py`, `src/models/__init__.py`, `src/auth/` |
| Rate limiting | slowapi + Redis (optional; falls back if Redis down) | `src/auth/dependencies.py` |
| LLMs | OpenAI, Anthropic, Gemini, Ollama, etc. | `src/llm/` |
| Infra (GCP) | Terraform modules | `terraform/` |

**Import layout:** Run with `PYTHONPATH` pointing at the project root (see [AGENTS.md](../AGENTS.md)).

---

## 3. Project structure (top level)

| Path | Role |
|------|------|
| `src/` | Application code: API, retrieval, embeddings, knowledge graph, LLM, feedback, DB access, CLI commands |
| `config/` | YAML configuration for embeddings, retrieval, LLM, training |
| `scripts/` | One-off and maintenance scripts (indexing, migrations helpers, etc.) |
| `tests/` | Pytest; `e2e/` and some integration tests need network—often skipped in CI/cloud |
| `scrapers/` | Scrapy projects / spiders for training data |
| `data/` | Local databases, training artifacts, checkpoints (many files gitignored) |
| `docs/` | Architecture, DB guides, scraper docs, **this SPEC**, AI-oriented rules |
| `terraform/` | Cloud Run, secrets, etc. |

**Main entrypoints**

- HTTP: `uvicorn src.api.server:app` (see README / AGENTS)
- CLI: `mist-cli` from editable install (`pyproject.toml` → `src.cli.main:app`)

---

## 4. Important components / files

| Path | Responsibility | Why it matters |
|------|----------------|----------------|
| `src/api/server.py` | FastAPI app, routes, startup, deps | Public API surface; wires RAG, feedback, auth |
| `src/api/schemas.py` | Pydantic request/response models | Contract for `/query`, `/clarify`, feedback bodies |
| `src/retrieval/conversational_rag.py` | Orchestrates query/clarify flow | Domain entry for “one diagnostic conversation” |
| `src/retrieval/enhanced_retriever.py` | Multi-stage retrieval (vector → rerank → KG → rank) | Core recommendation pipeline |
| `src/retrieval/chroma_store.py` | Chroma persistence / queries | Vector search backend |
| `src/knowledge/graph_query.py` | KG scoring over NetworkX graph | Connects ISTA relationships to ranking |
| `src/llm/provider.py` + clients | LLM selection / calls | Clarification and expansion depend on this |
| `src/feedback/collector.py` | SQLite feedback persistence | Ratings, outcomes, corrections |
| `src/auth/dependencies.py` | IAP identity, tiers, rate limits, `get_current_user` | Protects `/query` and most routes |
| `src/auth/routes.py` | Registration / auth HTTP routes | User lifecycle beside IAP |
| `src/paths.py` | Resolved paths to DBs and configs | Avoid hardcoding data locations |
| `src/cli/main.py` | Typer entry | Operational commands for ops and dev |

---

## 5. Domain & data model

**Concepts**

- **Diagnostic session (in-memory / RAG):** `SessionManager` holds turn state for clarification (`src/retrieval/session_manager.py`).
- **Feedback session (SQLite):** `FeedbackSession` / `MistFeedback` in `src/database/schema.py`—persistent telemetry for learning and analytics.
- **User (Postgres):** `User`, `Role`, `RateLimitTier` in `src/models/__init__.py`—identity, RBAC, tiered rate limits for deployed API.
- **Repair guide:** Identified by ISTA-style IDs; text and metadata live in Chroma (indexed from ISTA/XML pipelines—see indexing scripts and `docs/`).

**Where logic lives**

- **Transport:** FastAPI handlers in `server.py`—thin; validate and call services.
- **Orchestration:** `ConversationalRAG`, `EnhancedRetriever`, rankers/rerankers.
- **Data access:** `src/database/*` (ISTA SQLite, feedback SQLite, Postgres), `src/retrieval/chroma_store.py`.

---

## 6. Control & data flow

### Example: `POST /query`

1. **FastAPI** `query()` in `src/api/server.py` → `Depends(get_current_user)` (IAP + DB user + tier).
2. **`ConversationalRAG.query()`** (`conversational_rag.py`) → session create/restore, optional ambiguity + clarification.
3. **`EnhancedRetriever`** → encodes query, **Chroma** similarity, **Reranker**, **KnowledgeGraphQuery** scoring, **Ranker** fusion (and feedback signals where configured).
4. Response mapped to **`QueryResponse`** in `schemas.py`.

### Example: `POST /feedback/rating`

1. `server.py` → **`FeedbackCollector.save_session`** → **SQLite** via `src/database/connection.py` + ORM in `schema.py`.

**Clarify path:** `POST /clarify` → `ConversationalRAG.clarify()` → `QueryExpander` / session update → same retrieval stack.

---

## 7. Cross-cutting concerns

- **Configuration:** YAML under `config/`; paths via `src/paths.py`. Env vars for secrets and overrides (API keys, `CHROMA_DB_*`, `DATABASE_URL`, `REDIS_URL`, `RATE_LIMIT_IP_FALLBACK`, `ISTA_DB_PATH`, etc.).
- **Auth (deployed):** Google IAP headers (`X-Goog-Authenticated-User-*`) + user rows in Postgres; **`/query` and `/clarify`** use **slowapi** with a dynamic limit from the user’s `RateLimitTier` (sync DB lookup by rate-limit key in `tier_limit_for_ratelimit_key`, `src/auth/dependencies.py`). Requests without an IAP email fall back to an IP bucket; override with **`RATE_LIMIT_IP_FALLBACK`** (default `1000/minute` for local/tests).
- **Security middleware:** `src/api/security.py` (e.g. API keys where configured).
- **Logging:** Standard `logging` module; prefer module-level `logger = logging.getLogger(__name__)`.

**Invariant:** Do not drop or delete production user/feedback/ISTA data casually; migrations and destructive ops need explicit intent (see user rules).

---

## 8. Testing & quality

- **Runner:** `pytest` — from repo root with `PYTHONPATH` set to project root (see AGENTS).
- **Layout:** `tests/` with `unit/`, `integration/`, `e2e/`; many tests are file-named `test_*.py`.
- **Known gaps:** AGENTS notes pre-existing failures in some config/session/retrieval/prompt tests; model-heavy tests download weights.
- **Expectation:** New behavior should get tests near the module under test; mirror existing patterns in `tests/test_*`.

---

## 9. Constraints & gotchas

| Topic | Detail |
|-------|--------|
| **ISTA DB** | Multi-GB `DiagDocDb_Decrypted.sqlite` usually **not** in git; `/query` quality depends on it + indexed Chroma |
| **Chroma** | Cloud credentials (`CHROMA_DB_API_KEY`, `CHROMA_DB_TENANT`) required for production-like retrieval |
| **Postgres** | `src/database/pg_connection.py` requires **`DATABASE_URL`** at import time for processes that load the API stack |
| **Redis** | Used for rate limiting; connection failure logs a warning and may degrade behavior—check `src/auth/dependencies.py` |
| **Dual feedback meaning** | “Session” in RAG ≠ row in SQLite feedback DB; link by `session_id` string |
| **Terraform state** | `terraform/*.tfstate` may contain environment-specific values—treat as sensitive in real deployments |

When architecture, endpoints, env vars, or major modules change, update **this file** and [REPO_RULES_FOR_AI.md](REPO_RULES_FOR_AI.md) alongside [ARCHITECTURE.md](ARCHITECTURE.md).

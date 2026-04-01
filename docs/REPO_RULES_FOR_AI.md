# Repository rules for AI assistants (MIST)

Read this **together with** [SPEC.md](SPEC.md), [agent.md](agent.md), and [ARCHITECTURE.md](ARCHITECTURE.md). Cursor also loads `.cursor/rules/*.mdc` (context-first, pre/post gates, update-docs).

---

## 1. General principles

- **Prefer existing patterns** over new frameworks, folders, or abstractions unless the task explicitly requires a new layer.
- **Verify before importing** — open the file or grep; do not assume symbols from memory.
- **Minimal diffs** — touch only what the task needs; avoid drive-by refactors and unrelated formatting.
- **Python 3.12** — match existing typing and style in the touched files.
- **Data safety** — never delete or drop user data, ISTA data, or feedback tables unless the user explicitly requests it and understands the impact.

---

## 2. Architecture & file placement

| Kind of change | Where it goes |
|----------------|----------------|
| New HTTP route or middleware | `src/api/server.py` or a new router under `src/api/` included from `server.py` |
| Pydantic API models | `src/api/schemas.py` |
| Auth / IAP / tiers / rate limit keys | `src/auth/dependencies.py`, `src/auth/routes.py`; ORM models in `src/models/` |
| Postgres session/engine | `src/database/pg_connection.py` |
| Retrieval pipeline stages | `src/retrieval/` (orchestrators: `enhanced_retriever.py`, `conversational_rag.py`) |
| Embedding backends | `src/embeddings/` |
| LLM calls / prompts | `src/llm/` (e.g. `prompt_templates.py`, provider clients) |
| BMW ISTA read-only access | `src/database/ista_db.py`, `xml_content.py`, `fault_code_mapping.py` |
| Feedback persistence (SQLite) | `src/feedback/collector.py`, `src/database/schema.py` |
| CLI commands | New module under `src/commands/`, register in `src/cli/main.py` |
| User-facing config | `config/*.yaml` + document in README / ARCHITECTURE / SPEC |
| One-off batch jobs | `scripts/` |

---

## 3. Naming & style

- Follow **existing module naming** in the same directory (`*_retriever.py`, `*_encoder.py`, etc.).
- Use **module-level loggers**: `logger = logging.getLogger(__name__)`.
- Keep **FastAPI dependencies** as small functions in `server.py` or dedicated deps modules if the file grows.
- **Tests:** `tests/test_<module>.py` or `tests/unit/test_<feature>.py` consistent with neighbors.

---

## 4. Data & API contracts

- **Public JSON contracts** live in `src/api/schemas.py` — extend carefully; prefer optional fields for backward compatibility.
- **Environment variables** that affect behavior should be mentioned in README, AGENTS, or SPEC when added.
- **Two databases:** Postgres (users/auth) vs SQLite (feedback) vs ISTA SQLite — do not conflate them in code or docs.

---

## 5. Testing

- Run **unit/integration** tests with `PYTHONPATH` set to repo root (see [AGENTS.md](../AGENTS.md)).
- Skip **`tests/e2e`** in unattended/cloud runs unless explicitly requested.
- If you change retrieval, schemas, or RAG flow, consider **updating or adding** tests under `tests/` that mirror similar components.
- **Pre-commit:** with `pip install -e ".[dev]"` and `lefthook install`, staged Python files are checked with **Ruff** (`lefthook.yml`). Fix reported issues before committing (or set `LEFTHOOK=0` only when intentional).

---

## 6. Safety & high-risk areas

- **`.pi/`** — Pi editor extensions are **local-only** (directory is gitignored); do not add them to this repo.
- **Secrets** — never commit real API keys, `DATABASE_URL` values, or `terraform.tfvars`. Use env vars, Secret Manager, and the tracked `terraform.tfvars.example` / placeholder `k8s/secrets.yaml`. If credentials ever landed in git history, **rotate them** in the provider; removing files from the current tree does not erase past commits.
- **`terraform/`** — infra and secrets; do not casually rewrite production bindings.
- **Indexing / migration scripts** — can overwrite or corrupt local indices; understand flags before running.
- **`src/database/pg_connection.py`** — importing the API stack requires valid `DATABASE_URL` configuration in many setups.
- **Large model downloads** — tests and encoders may pull Hugging Face weights; avoid running unbounded in CI without caching.

---

## 7. When to update documentation

- After substantive edits, apply the **post-agent checklist** in `.cursor/rules/post-agent-context.mdc` and [update-docs-after-changes.mdc](../.cursor/rules/update-docs-after-changes.mdc).
- If you change **routes, env vars, module layout, or pipeline stages**, update **[SPEC.md](SPEC.md)** and relevant sections of **[ARCHITECTURE.md](ARCHITECTURE.md)**.
- If you add new **doc files**, link them from **[agent.md](agent.md)**.

---

## 8. If uncertain

- Open a **similar existing implementation** in the same package and mirror it.
- Ask the user to confirm **env assumptions** (e.g. local SQLite-only vs full Postgres + IAP) before designing auth or deployment changes.

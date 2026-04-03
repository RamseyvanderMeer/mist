# MIST — condensed context

One-page spine. For path-precise contracts use **[core/SPEC.md](core/SPEC.md)**. For narrative design use **[core/ARCHITECTURE.md](core/ARCHITECTURE.md)**.

## What it is

**MIST** (Multi-modal Intelligent Service Technician) is a **Python 3.12** service that recommends BMW repair guides from **fault codes**, **OBD data**, and **natural language**. It combines **ChromaDB vector search**, **re-ranking**, **ISTA knowledge-graph signals**, and **LLMs** for clarification and prompts.

**Runtime shape:** one **FastAPI** app (`src/api/server.py`), **Typer CLI** (`mist-cli` → `src/cli/main.py`), scripts under `scripts/`, optional **Scrapy** under `scrapers/`, **Terraform** for GCP.

## Main flows

| Flow | Entry | Notes |
|------|--------|------|
| Query / clarify | `POST /query`, `POST /clarify` | `ConversationalRAG` → `EnhancedRetriever` → Chroma + rerank + KG + ranker |
| Auth (deployed) | `src/auth/dependencies.py` | IAP headers + JWT verification; Postgres users/tiers; slowapi + Redis |
| Feedback | `POST /feedback/*` | SQLite app DB (`src/feedback/`, `src/database/schema.py`) |
| Indexing | `scripts/index_repair_guides.py` | ISTA + xmlvalueprimitive → embeddings → Chroma |

## Stack (short)

| Concern | Where |
|---------|--------|
| API | `src/api/server.py`, `src/api/schemas.py` |
| RAG orchestration | `src/retrieval/conversational_rag.py`, `enhanced_retriever.py` |
| Vectors | `src/retrieval/chroma_store.py`, `config/retrieval_config.yaml` |
| ISTA / XML text | `src/database/ista_db.py`, `src/database/xml_content.py` |
| Embeddings | `src/embeddings/` |
| LLM | `src/llm/` |
| Config / paths | `config/*.yaml`, `src/paths.py` |

## Topic folders (dig deeper)

| Topic | Folder |
|--------|--------|
| Spec, architecture, AI placement rules | [core/](core/) |
| BMW SQLite, ISTA layout | [data/](data/) |
| Retrieval behavior, mismatch notes | [retrieval/](retrieval/) |
| Prompt / diagnostic framing | [llm/](llm/) |
| Scrapers, scraping prompt | [scraping/](scraping/) |
| Training plans, data optimization | [training/](training/) |
| IAP, proxy, production access | [deploy/](deploy/) |
| Recent `main` changes | [changelog/](changelog/) |
| Structure proposals | [planning/](planning/) |

## Outside this tree

| Location | Role |
|----------|------|
| [AGENTS.md](../../AGENTS.md) | Cursor/cloud setup, venv, pytest, hooks |
| [README.md](../../README.md) | Human-oriented setup |
| [data/databases/README.md](../../data/databases/README.md) | DB filenames on disk |
| [data/training/README.md](../../data/training/README.md) | Training CSVs |
| [scrapers/PRODUCTION_READINESS_ANALYSIS.md](../../scrapers/PRODUCTION_READINESS_ANALYSIS.md) | Scraper gaps |

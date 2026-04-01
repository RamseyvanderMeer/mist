# MIST Architecture

MIST (Multi-modal Intelligent Service Technician) is an AI-powered automotive diagnostic system that maps fault codes and OBD live data to repair guide recommendations.

## High-Level Flow

```
Fault Codes + OBD Data → Multi-Modal Encoding → Vector Search (ChromaDB) →
Re-ranking → Knowledge Graph Filtering → Combined Scoring →
Recommendations (with optional clarification)
```

## Core Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **FaultCodeEncoder** | `src/embeddings/fault_code_encoder.py` | E5-Mistral-7B-Instruct, projects to 768/1024 dim |
| **OBDDataEncoder** | `src/embeddings/obd_data_encoder.py` | Encodes OBD sensor JSON |
| **MultiModalEncoder** | `src/embeddings/multimodal_encoder.py` | Cross-attention fusion |
| **ChromaVectorStore** | `src/retrieval/chroma_store.py` | ChromaDB Cloud for repair guide embeddings |
| **Reranker** | `src/retrieval/reranker.py` | Local cross-encoder or Cohere API |
| **Ranker** | `src/retrieval/ranker.py` | Combines embedding, KG, feedback, recency |
| **KnowledgeGraph** | `src/knowledge/` | NetworkX graph from BMW ISTA DB |
| **ConversationalRAG** | `src/retrieval/conversational_rag.py` | Orchestrator, clarification, session management |
| **LLM Providers** | `src/llm/` | OpenAI, Anthropic, Gemini, Ollama |
| **FeedbackCollector** | `src/feedback/collector.py` | SQLite feedback storage |
| **EmbeddingTrainer** | `src/learning/` | Contrastive fine-tuning from feedback |

## Configuration

YAML configs in `config/`:

- `embedding_config.yaml` – Model settings, projection dim, fusion
- `retrieval_config.yaml` – ChromaDB, retrieval params, ranking weights, clarification
- `llm_config.yaml` – LLM provider and model selection
- `training_config.yaml` – Fine-tuning hyperparameters

## API, auth, and data stores

- **FastAPI app:** `src/api/server.py` — registers routes, CORS, rate limiting, and includes `src/auth/routes.py`.
- **Authentication (typical cloud setup):** Google IAP headers (`X-Goog-Authenticated-User-Email`, `X-Goog-Authenticated-User-Id`) with user rows in **PostgreSQL** (`DATABASE_URL`, models in `src/models/`, session in `src/database/pg_connection.py`). Tier-based rate limits use **slowapi** and optionally **Redis** (`REDIS_URL`). See `src/auth/dependencies.py`.
- **Feedback and diagnostic telemetry:** **SQLite** at `data/databases/mist_data.db` (via `src/paths.py` / `FeedbackCollector`) — separate from Postgres users.
- **ISTA diagnostic content:** Large local SQLite (`DiagDocDb_Decrypted.sqlite`) for knowledge graph and grounding — see [DATABASE.md](DATABASE.md).

For a path-indexed summary and gotchas, see [SPEC.md](SPEC.md).

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Process fault codes + OBD data |
| `/clarify` | POST | Provide clarification responses |
| `/auth/*` | various | Registration and auth (see `src/auth/routes.py`) |
| `/feedback/rating` | POST | Submit rating |
| `/feedback/outcome` | POST | Submit repair outcome |
| `/feedback/correction` | POST | Submit corrections |
| `/feedback/statistics` | GET | Feedback stats (admin) |
| `/feedback/{session_id}` | GET | Session feedback (admin) |
| `/health` | GET | Health check |

## Getting Started

```bash
# Setup
pip install -r requirements.txt
python scripts/run_migrations.py

# Build knowledge graph (requires DiagDocDb_Decrypted.sqlite)
python scripts/build_knowledge_graph.py

# Index repair guides (requires ChromaDB Cloud: CHROMA_DB_API_KEY, CHROMA_DB_TENANT)
python scripts/index_repair_guides.py

# Run API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

## Path Management

`src/paths.py` provides `get_paths()` for database and data paths. Uses `data/databases/` by default; override with `ISTA_DB_PATH` or `MIST_DATABASE_DIR`.

## Further Reading

- [DATABASE.md](DATABASE.md) – BMW ISTA database overview
- [ISTA_DATABASE_GUIDE.md](ISTA_DATABASE_GUIDE.md) – Document hierarchy, Process Analysis
- [README.md](../README.md) – Setup and usage

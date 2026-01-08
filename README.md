# MIST - Multi-modal Intelligent Service Technician

AI-powered automotive diagnostic system that maps fault codes and live OBD data to precise repair guide recommendations.

## Overview

MIST combines:
- **Multi-Modal Embeddings**: Fault codes (text) + OBD sensor data (structured)
- **Conversational RAG**: LLM-guided clarification for ambiguous cases
- **Self-Improving**: Learns from feedback and repair outcomes
- **Knowledge Graph Integration**: Leverages BMW diagnostic database relationships

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Migrate Databases** (if not already done)
   ```bash
   python scripts/migrate_databases.py
   ```

4. **Build Knowledge Graph**
   ```bash
   python scripts/build_knowledge_graph.py
   ```

5. **Index Repair Guides**
   ```bash
   python scripts/index_repair_guides.py
   ```

6. **Start API Server**
   ```bash
   python -m src.api.server
   # Or: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```

## Project Structure

```
mist/
├── src/              # Source code
│   ├── embeddings/   # Multi-modal encoders
│   ├── retrieval/    # RAG components
│   ├── knowledge/    # Knowledge graph
│   ├── llm/          # LLM providers
│   ├── feedback/     # Feedback system
│   ├── learning/     # Self-improvement
│   └── api/          # FastAPI server
├── config/           # YAML configuration files
├── scripts/          # Utility scripts
├── data/             # Databases, vector store, checkpoints
├── tests/            # Test suite
└── docs/             # Documentation
```

## Documentation

- [QUICK_START.md](QUICK_START.md): Quick reference guide
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): Detailed implementation plan
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Architecture overview
- [docs/DATABASE.md](docs/DATABASE.md): Database documentation
- [docs/API.md](docs/API.md): API documentation (to be created)

## Configuration

Edit YAML files in `config/`:
- `embedding_config.yaml`: Embedding model settings
- `retrieval_config.yaml`: Retrieval parameters
- `llm_config.yaml`: LLM provider settings
- `training_config.yaml`: Training hyperparameters

## Features

- **Multi-Stage Retrieval**: Vector search → Re-ranking → KG filtering → Combined scoring
- **Conversational Clarification**: LLM-guided questions for ambiguous cases
- **Self-Improving**: Continuous learning from user feedback
- **Knowledge Graph**: Exploits BMW diagnostic relationships for explainable recommendations

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
- GPU recommended for embedding generation (optional)

## License

[Add license information]

## Contributing

[Add contributing guidelines]

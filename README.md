# MIST - Multi-modal Intelligent Service Technician

AI-powered automotive diagnostic system that maps fault codes and live OBD data to precise repair guide recommendations.

Demo: [mist-expo.vercel.app/sign-in](https://mist-expo.vercel.app/sign-in)

## Overview

MIST combines:
- **Multi-Modal Embeddings**: Fault codes (text) + OBD sensor data (structured)
- **Conversational RAG**: LLM-guided clarification for ambiguous cases
- **Self-Improving**: Learns from feedback and repair outcomes
- **Knowledge Graph Integration**: Leverages BMW diagnostic database relationships

## Setup

### Prerequisites

- Python 3.12+ (recommended)
- Virtual environment (recommended)
- GPU recommended for training and embedding generation (optional)

### Installation Steps

1. **Clone the repository** (if not already done)
   ```bash
   cd mist
   ```

2. **Create and activate virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .  # Editable install: enables `mist-cli` and importable packages
   ```

   Optional — **lint + git hooks** (Ruff + Lefthook):

   ```bash
   pip install -e ".[dev]"
   lefthook install
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root (if `.env.example` exists, copy it):
   ```bash
   # If .env.example exists:
   cp .env.example .env
   
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_openai_key
   # ANTHROPIC_API_KEY=your_anthropic_key
   # COHERE_API_KEY=your_cohere_key  # Optional, for re-ranking
   
   # ChromaDB Cloud (required for vector store, see step 7):
   # CHROMA_DB_API_KEY=your-chromadb-api-key
   # CHROMA_DB_TENANT=your-chromadb-tenant-id
   ```

5. **Run database migrations**
   ```bash
   mist-cli migrate
   # Or: python scripts/run_migrations.py
   ```

6. **Build knowledge graph** (one-time setup)
   ```bash
   mist-cli build-kg
   # Or: python scripts/build_knowledge_graph.py
   ```

7. **Configure ChromaDB Cloud** (required for vector store)
   - Set in `.env`:
     ```bash
     CHROMA_DB_API_KEY=your-api-key
     CHROMA_DB_TENANT=your-tenant-id
     ```
   - See `config/retrieval_config.yaml` for collection and database settings.

8. **Index repair guides** (one-time setup, may take a while)
   ```bash
   mist-cli index
   # Or: python -m src.cli.main index
   # Or: python scripts/index_repair_guides.py
   ```

   Use `mist-cli` (not `mist`) to avoid conflict with the npm package.

## Training

After collecting feedback data through the API, you can train the embeddings to improve retrieval performance.

### Training Embeddings

Train embeddings using contrastive learning on collected feedback:

```bash
# Basic training with default configs
mist-cli train

# With custom config files
mist-cli train --config config/training_config.yaml --embedding-config config/embedding_config.yaml

# Resume from checkpoint
mist-cli train --resume data/embeddings/checkpoints/checkpoint_epoch_5.pt

# Specify device (auto, cuda, or cpu)
mist-cli train --device cuda

# Set logging level
mist-cli train --log-level DEBUG
```

### Training Configuration

Edit `config/training_config.yaml` to adjust:
- Learning rate
- Batch size
- Number of epochs
- Checkpoint frequency
- Loss function parameters

The training script will:
1. Load feedback data from the feedback database
2. Create positive/negative pairs for contrastive learning
3. Fine-tune the multi-modal encoder using InfoNCE loss
4. Save checkpoints periodically
5. Save the final trained model

## Demo Usage

### Option 1: Using the Python API

```python
from src.retrieval.conversational_rag import ConversationalRAG

# Initialize the RAG system
rag = ConversationalRAG()

# Query with fault codes and OBD data
response = rag.query(
    fault_codes=["P0300", "Random/Multiple Cylinder Misfire"],
    obd_data={
        "engine_rpm": 2500,
        "coolant_temp": 95,
        "throttle_position": 45,
        "maf_sensor": 12.5
    },
    vehicle_context={
        "model": "335i",
        "year": 2011,
        "engine": "N55"
    }
)

# Check if clarification is needed
if response['needs_clarification']:
    print("Clarification questions:")
    for question in response['clarification_questions']:
        print(f"  - {question}")
    
    # Provide responses and get refined recommendations
    refined = rag.clarify(
        session_id=response['session_id'],
        responses=["Yes, the check engine light is flashing", "No unusual sounds"]
    )
    recommendations = refined['recommendations']
else:
    recommendations = response['recommendations']

# Display recommendations
print(f"\nFound {len(recommendations)} recommendations:")
for rec in recommendations:
    print(f"\n{rec['title']}")
    print(f"  Score: {rec['score']:.3f}")
    print(f"  Procedure: {rec['procedure_name']}")
```

### Option 2: Using the HTTP API

1. **Start the API server**
   ```bash
   python -m src.api.server
   # Or with uvicorn:
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```

2. **Query the API** (using curl or any HTTP client)

   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Query with fault codes and OBD data
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "fault_codes": ["P0300", "Random/Multiple Cylinder Misfire"],
       "obd_data": {
         "engine_rpm": 2500,
         "coolant_temp": 95,
         "throttle_position": 45
       },
       "vehicle_context": {
         "model": "335i",
         "year": 2011
       }
     }'
   ```

   Example response:
   ```json
   {
     "recommendations": [
       {
         "id": "guide_123",
         "title": "Misfire Diagnosis - Cylinder 1",
         "procedure_name": "Misfire Detection",
         "procedure_id": "PROC_001",
         "score": 0.92,
         "text": "Check ignition coil and spark plug..."
       }
     ],
     "needs_clarification": false,
     "clarification_questions": null,
     "session_id": "session_abc123",
     "query_text": "Random/Multiple Cylinder Misfire P0300"
   }
   ```

3. **Handle clarification** (if needed)

   ```bash
   curl -X POST http://localhost:8000/clarify \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session_abc123",
       "responses": [
         "Yes, the check engine light is flashing",
         "No unusual sounds"
       ]
     }'
   ```

4. **Submit feedback**

   ```bash
   # Submit rating
   curl -X POST http://localhost:8000/feedback/rating \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session_abc123",
       "rating": 5,
       "selected_guide": "guide_123"
     }'
   
   # Submit repair outcome
   curl -X POST http://localhost:8000/feedback/outcome \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session_abc123",
       "outcome": "success"
     }'
   ```

5. **View API documentation**

   Once the server is running, visit:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Option 3: Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Query
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "fault_codes": ["P0300"],
        "obd_data": {"engine_rpm": 2500, "coolant_temp": 95},
        "vehicle_context": {"model": "335i", "year": 2011}
    }
)
result = response.json()

print(f"Session ID: {result['session_id']}")
print(f"Recommendations: {len(result['recommendations'])}")

# Submit feedback
requests.post(
    f"{BASE_URL}/feedback/rating",
    json={
        "session_id": result['session_id'],
        "rating": 5,
        "selected_guide": result['recommendations'][0]['id']
    }
)
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
├── apps/mist-expo/   # Expo + Tamagui client (web, iOS, Android) — see apps/mist-expo/README.md
├── config/           # YAML configuration files
├── scripts/          # Utility scripts
├── data/             # Databases, vector store, checkpoints
├── tests/            # Test suite
└── docs/             # Documentation
```

## Documentation

Topic-based context for humans and AI agents lives under **`docs/context/`**. Start at **[docs/context/README.md](docs/context/README.md)** (navigation tree).

- [docs/context/OVERVIEW.md](docs/context/OVERVIEW.md): One-page condensed spine
- [docs/context/core/SPEC.md](docs/context/core/SPEC.md): Path-grounded spec and flows
- [docs/context/core/ARCHITECTURE.md](docs/context/core/ARCHITECTURE.md): Architecture and getting started
- [docs/context/data/DATABASE.md](docs/context/data/DATABASE.md): BMW ISTA database overview

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

- Python 3.12+ (recommended)
- Virtual environment (recommended)
- See `requirements.txt` for dependencies
- GPU recommended for training and embedding generation (optional)

## License

[Add license information]


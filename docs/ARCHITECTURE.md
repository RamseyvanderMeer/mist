# MIST Architecture Overview

## System Architecture

MIST (Multi-modal Intelligent Service Technician) is an AI-powered automotive diagnostic system that maps fault codes and OBD live data to repair guide recommendations.

## High-Level Flow

```
Fault Codes + OBD Data → Multi-Modal Encoding → Vector Search → 
Re-ranking → Knowledge Graph Filtering → Combined Scoring → 
Recommendations (with optional clarification)
```

## Core Components

### 1. Multi-Modal Embeddings
- **FaultCodeEncoder**: Encodes fault code text using E5-Mistral-7B-Instruct
- **OBDDataEncoder**: Encodes structured OBD sensor data
- **MultiModalEncoder**: Cross-attention fusion combining both modalities

### 2. Retrieval Pipeline
- **VectorStore**: Qdrant-based vector database for repair guide embeddings
- **Reranker**: Cross-encoder or Cohere API for fine-grained relevance scoring
- **Ranker**: Combines multiple signals (embedding, KG, feedback, recency)

### 3. Knowledge Graph
- **GraphBuilder**: Extracts relationships from BMW diagnostic database
- **GraphQuery**: Path finding and relationship reasoning

### 4. Conversational RAG
- **ConversationalRAG**: Main orchestrator for multi-turn conversations
- **QueryExpander**: Expands queries with user clarification responses
- **LLM Providers**: OpenAI, Anthropic, or local Ollama

### 5. Self-Improvement
- **FeedbackCollector**: Stores user feedback and repair outcomes
- **EmbeddingTrainer**: Fine-tunes embeddings using contrastive learning
- **ActiveLearning**: Identifies uncertain cases for human review

## Data Flow

See [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for detailed architecture diagrams and specifications.

## Configuration

All configuration is YAML-based:
- `config/embedding_config.yaml`: Embedding model settings
- `config/retrieval_config.yaml`: Retrieval parameters
- `config/llm_config.yaml`: LLM provider settings
- `config/training_config.yaml`: Training hyperparameters

## Path Management

Centralized path management via `src/paths.py`:
- Auto-detects MIST root directory
- Supports environment variable overrides
- Provides fallback to old database locations

## API

FastAPI server provides REST endpoints:
- `POST /query`: Process fault codes and OBD data
- `POST /clarify`: Provide clarification responses
- `POST /feedback/*`: Submit feedback
- `GET /feedback/statistics`: Get feedback stats

## For More Details

- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md): Complete implementation specifications
- [QUICK_START.md](../QUICK_START.md): Quick start guide
- [API.md](API.md): API documentation (to be created)

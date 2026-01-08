# MIST Enhanced Mapping Layer - Implementation Plan

## Executive Summary

This document outlines the design and implementation plan for an enhanced AI-driven mapping layer that transforms fault codes and live OBD data into precise repair guide recommendations. The system builds upon the existing MIST architecture while incorporating cutting-edge AI techniques for self-improvement, conversational clarification, and continuous learning from real-world diagnostic experiences.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Cutting-Edge AI Techniques](#3-cutting-edge-ai-techniques)
4. [Component Specifications](#4-component-specifications)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Integration Points](#6-integration-points)
7. [Self-Improvement Mechanisms](#7-self-improvement-mechanisms)
8. [Database Schema Extensions](#8-database-schema-extensions)

---

## 1. System Overview

### 1.1 Problem Statement

The current BMW ISTA reverse engineering kit provides access to diagnostic databases and repair procedures, but lacks an intelligent mapping layer that can:
- Combine fault codes with live OBD sensor data for context-aware diagnosis
- Ask clarifying questions when diagnostic ambiguity exists
- Learn from repair outcomes to improve future recommendations
- Handle multi-modal inputs (text fault codes + structured sensor data)

### 1.2 Solution Architecture

The enhanced mapping layer will:
1. **Multi-Modal Encoding**: Transform fault codes (text) and OBD live data (structured) into unified embeddings
2. **Conversational RAG**: Use LLM-guided clarification questions to refine ambiguous cases
3. **Self-Improving Retrieval**: Continuously learn from user feedback and repair outcomes
4. **Knowledge Graph Integration**: Leverage BMW diagnostic database relationships for explainable recommendations

### 1.3 Key Innovations

- **Hybrid Embedding Architecture**: Combines transformer-based text encoders with structured data encoders using cross-attention fusion
- **Active Learning Pipeline**: Identifies uncertain cases for human review and feedback collection
- **Reward-Based Fine-Tuning**: Uses RLHF (Reinforcement Learning from Human Feedback) to improve embedding quality
- **Multi-Stage Retrieval**: Combines vector search, re-ranking, and knowledge graph constraints

---

## 2. Architecture Design

### 2.1 High-Level System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Layer                                  │
│  Fault Codes (Text) + OBD Live Data (JSON) + Vehicle Context    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Modal Embedding Layer                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │ Fault Code   │   │ OBD Data     │   │ Cross-Attention  │     │
│  │ Encoder      │   │ Encoder      │   │ Fusion Layer     │     │
│  │ (E5-Mistral) │   │ (Structured) │   │                  │     │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘     │
│         │                  │                    │               │
│         └───────────────-──┴────────────────────┘               │
│                            │                                    │
│                            ▼                                    │
│                    Unified Embedding (768-dim)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Stage Retrieval Pipeline                      │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Stage 1: Vector Search (Qdrant)                          │    │
│  │   - Initial retrieval: Top-K=100                         │    │
│  │   - Cosine similarity                                    │    │
│  └──────────────────────┬───────────────────────────────────┘    │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Stage 2: Cross-Encoder Re-ranking (Cohere Rerank 3)      │    │
│  │   - Re-rank top-K=50                                     │    │
│  │   - Query-document relevance scoring                     │    │
│  └──────────────────────┬───────────────────────────────────  ┘  │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Stage 3: Knowledge Graph Filtering                       │    │
│  │   - Path scoring: Fault → ECU → Diagnostic → Repair      │    │
│  │   - Relationship weighting                               │    │
│  └──────────────────────┬───────────────────────────────────┘    │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Stage 4: Combined Scoring                                │    │
│  │   - Embedding similarity (40%)                           │    │
│  │   - Re-rank score (30%)                                  │    │
│  │   - KG path score (20%)                                  │    │
│  │   - Feedback score (10%)                                 │    │
│  └──────────────────────┬───────────────────────────────────┘    │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Ambiguity Detection & Clarification                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Ambiguity Check:                                         │   │
│  │   - Top score < threshold (0.65)                         │   │
│  │   - Score variance < threshold (0.02)                    │   │
│  │   - Missing critical OBD parameters                      │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                       │
│         ┌───────────────┴───────────────┐                       │
│         │                               │                       │
│         ▼                               ▼                       │
│  ┌──────────────┐            ┌──────────────────┐               │
│  │ High         │            │ Low Confidence   │               │
│  │ Confidence   │            │ → Generate       │               │
│  │ → Return     │            │ Clarification    │               │
│  │ Results      │            │ Questions (LLM)  │               │
│  └──────────────┘            └────────┬─────────┘               │
│                                       │                         │
│                                       ▼                         │
│                              ┌──────────────────┐               │
│                              │ User Responses   │               │
│                              └────────┬─────────┘               │
│                                       │                         │
│                                       ▼                         │
│                              ┌──────────────────┐               │
│                              │ Query Expansion  │               │
│                              │ (LLM-based)      │               │
│                              └────────┬─────────┘               │
│                                       │                         │
│                                       ▼                         │
│                              ┌──────────────────┐               │
│                              │ Re-process Query │               │
│                              └──────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Layer                                 │
│  Ranked Repair Guide Recommendations + Confidence Scores        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Multi-Modal Embedding System

**Fault Code Encoder:**
- **Model**: `intfloat/e5-mistral-7b-instruct` (state-of-the-art instruction-tuned embeddings)
- **Input**: Fault code descriptions (e.g., "Random/Multiple Cylinder Misfire Detected")
- **Output Dimension**: 4096 (native) → projected to 768
- **Rationale**: E5-Mistral provides superior semantic understanding compared to sentence-transformers

**OBD Data Encoder:**
- **Architecture**: Structured neural network with attention mechanism
- **Input**: Normalized OBD parameters (RPM, temperature, pressure, etc.)
- **Features**: 
  - Temporal patterns (if multiple readings available)
  - Parameter relationships (e.g., RPM vs throttle position)
  - Anomaly detection (values outside normal ranges)
- **Output Dimension**: 768

**Cross-Attention Fusion:**
- **Mechanism**: Bidirectional cross-attention between fault codes and OBD data
- **Architecture**: Multi-head attention (8 heads) with residual connections
- **Output**: Unified 768-dimensional embedding

#### 2.2.2 Retrieval Pipeline

**Vector Store (Qdrant):**
- **Collection**: `repair_guides_enhanced`
- **Vector Size**: 768
- **Distance Metric**: Cosine similarity
- **Metadata**: Procedure ID, fault codes, ECU category, vehicle compatibility

**Re-ranking Model:**
- **Model**: `Cohere Rerank 3` (or `cross-encoder/ms-marco-MiniLM-L-12-v2` for local)
- **Purpose**: Fine-grained relevance scoring
- **Input**: Query text + Document text pairs
- **Output**: Relevance scores (0-1)

**Knowledge Graph Integration:**
- **Graph**: NetworkX MultiDiGraph
- **Nodes**: Fault codes, ECUs, Diagnostic objects, Repair procedures
- **Edges**: Weighted relationships (affects_ecu, has_diagnostic, has_repair)
- **Path Scoring**: Shortest path algorithm with edge weight consideration

#### 2.2.3 Conversational RAG

**Clarification Question Generation:**
- **LLM**: GPT-4o or Claude 3.5 Sonnet
- **Prompt Template**: System role + Fault context + Top candidates
- **Output**: 1-3 clarifying questions
- **Parsing**: Extract questions from numbered/bulleted list

**Query Expansion:**
- **LLM**: Same as clarification
- **Input**: Original query + User responses
- **Output**: Expanded query with additional context

**Session Management:**
- **Storage**: SQLite database (`feedback_sessions`)
- **Tracking**: Multi-turn conversations, clarification history

#### 2.2.4 Self-Improvement System

**Feedback Collection:**
- **Types**: 
  - Explicit ratings (1-5)
  - Repair outcomes (success/failure/partial)
  - Conversation corrections
  - Selected guide tracking

**Reward Model:**
- **Architecture**: Neural network (768 → 512 → 256 → 1)
- **Training**: RLHF with feedback scores
- **Output**: Reward signal (0-1)

**Embedding Fine-Tuning:**
- **Method**: Contrastive learning
- **Loss**: InfoNCE loss with hard negative mining
- **Data**: Positive pairs (high ratings) vs Negative pairs (low ratings)
- **Checkpointing**: After each epoch

**Active Learning:**
- **Uncertainty Detection**: Entropy-based or score variance
- **Sampling**: Select cases with low confidence for human review
- **Batch Collection**: Prioritize uncertain cases for feedback

---

## 3. Cutting-Edge AI Techniques

### 3.1 Advanced Embedding Models

**E5-Mistral-7B-Instruct:**
- **Why**: Instruction-tuned embeddings provide better semantic understanding
- **Advantage**: Handles complex queries and domain-specific terminology
- **Implementation**: Use `sentence-transformers` wrapper or direct HuggingFace integration

**Alternative: BGE-M3 (Multilingual, Multimodal, Multi-Granularity):**
- **Why**: Supports multiple languages and granularities
- **Advantage**: Can handle multilingual BMW documentation

### 3.2 Re-ranking Techniques

**Cohere Rerank 3:**
- **Why**: State-of-the-art re-ranking model
- **Advantage**: Superior relevance scoring
- **Trade-off**: Requires API access (cost consideration)

**Local Alternative: Cross-Encoder Models:**
- **Models**: `cross-encoder/ms-marco-MiniLM-L-12-v2` or `BAAI/bge-reranker-base`
- **Advantage**: No API costs, runs locally
- **Trade-off**: Slightly lower performance

### 3.3 Self-Improvement Mechanisms

**Contrastive Learning:**
- **Method**: InfoNCE loss with hard negative mining
- **Implementation**: 
  - Positive pairs: Query + Relevant document (rating ≥ 4)
  - Negative pairs: Query + Irrelevant document (rating ≤ 2)
  - Hard negatives: Top-ranked but low-rated documents

**RLHF (Reinforcement Learning from Human Feedback):**
- **Reward Model**: Trained on explicit ratings and repair outcomes
- **Policy Gradient**: PPO (Proximal Policy Optimization) for embedding updates
- **Implementation**: Use `trl` library (Transformer Reinforcement Learning)

**Active Learning:**
- **Uncertainty Sampling**: Entropy-based selection
- **Query-by-Committee**: Multiple model ensemble for disagreement
- **Expected Model Change**: Select cases that would most improve the model

### 3.4 Knowledge Graph Enhancement

**Graph Neural Networks (GNNs):**
- **Model**: GraphSAGE or Graph Attention Network (GAT)
- **Purpose**: Learn node embeddings from graph structure
- **Advantage**: Captures complex relationships between faults and procedures

**Graph-Based Retrieval:**
- **Method**: Personalized PageRank or Random Walk with Restart
- **Purpose**: Find related procedures through graph traversal
- **Integration**: Combine with vector search for hybrid retrieval

---

## 4. Component Specifications

### 4.1 Project Structure

```
mist/
├── src/
│   ├── embeddings/
│   │   ├── fault_code_encoder.py      # E5-Mistral encoder
│   │   ├── obd_data_encoder.py        # Structured data encoder
│   │   ├── multimodal_encoder.py      # Cross-attention fusion
│   │   └── embedding_trainer.py       # Fine-tuning pipeline
│   ├── retrieval/
│   │   ├── vector_store.py           # Qdrant interface
│   │   ├── reranker.py                # Re-ranking module
│   │   ├── ranker.py                  # Combined scoring
│   │   ├── conversational_rag.py      # Main orchestrator
│   │   └── query_expander.py          # LLM-based expansion
│   ├── knowledge/
│   │   ├── graph_builder.py           # KG construction
│   │   ├── graph_query.py             # KG query interface
│   │   └── graph_embeddings.py        # GNN-based embeddings (optional)
│   ├── llm/
│   │   ├── provider.py                # Abstract base class
│   │   ├── openai_client.py           # OpenAI integration
│   │   ├── anthropic_client.py        # Anthropic integration
│   │   └── prompt_templates.py        # Prompt management
│   ├── feedback/
│   │   ├── collector.py               # Feedback storage
│   │   ├── analyzer.py                # Feedback analysis
│   │   └── reward_model.py            # RLHF reward model
│   ├── learning/
│   │   ├── active_learning.py         # Uncertainty sampling
│   │   └── contrastive_trainer.py    # Contrastive learning
│   └── api/
│       ├── server.py                   # FastAPI server
│       └── schemas.py                  # Pydantic models
├── config/
│   ├── llm_config.yaml                 # LLM provider settings
│   ├── embedding_config.yaml           # Embedding model config
│   ├── retrieval_config.yaml          # Retrieval parameters
│   └── training_config.yaml            # Training hyperparameters
├── scripts/
│   ├── build_knowledge_graph.py        # KG construction script
│   ├── index_repair_guides.py          # Vector store indexing
│   ├── train_embeddings.py             # Fine-tuning script
│   └── collect_feedback.py            # Feedback collection utility
├── data/
│   ├── databases/                      # BMW diagnostic databases
│   ├── knowledge_graph.graphml         # NetworkX graph file
│   ├── vector_store/                   # Qdrant data
│   ├── feedback/                       # Feedback database
│   └── embeddings/                     # Checkpoints
├── tests/
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_conversational_rag.py
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md              # This document
```

### 4.2 Key Components

#### 4.2.1 Fault Code Encoder

```python
class FaultCodeEncoder:
    """
    Encodes fault code descriptions using E5-Mistral-7B-Instruct.
    Provides superior semantic understanding compared to standard sentence-transformers.
    """
    def __init__(self, model_name="intfloat/e5-mistral-7b-instruct", device="auto"):
        self.model = SentenceTransformer(model_name, device=device)
        self.projection = nn.Linear(4096, 768)  # Project to 768-dim
    
    def encode(self, texts, normalize=True):
        """
        Encode fault code descriptions.
        
        Args:
            texts: Single string or list of strings
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            torch.Tensor: (batch_size, 768) embeddings
        """
        # E5-Mistral requires instruction prefix
        if isinstance(texts, str):
            texts = [texts]
        
        # Add instruction prefix for E5-Mistral
        prefixed_texts = [f"query: {text}" for text in texts]
        
        # Encode
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=True)
        
        # Project to 768-dim
        embeddings = self.projection(embeddings)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
```

#### 4.2.2 OBD Data Encoder

```python
class OBDDataEncoder(nn.Module):
    """
    Encodes structured OBD data with attention to parameter relationships.
    Handles temporal patterns if multiple readings available.
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=768):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Attention mechanism for parameter relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, obd_data):
        """
        Encode OBD data.
        
        Args:
            obd_data: Dict or list of dicts with OBD parameters
        
        Returns:
            torch.Tensor: (batch_size, 768) embeddings
        """
        # Normalize OBD parameters
        features = self.normalize_obd_data(obd_data)
        
        # Feature extraction
        x = self.feature_extractor(features)
        
        # Self-attention for parameter relationships
        x = x.unsqueeze(1)  # Add sequence dimension
        x_attn, _ = self.attention(x, x, x)
        x = x_attn.squeeze(1)
        
        # Output projection
        x = self.output_proj(x)
        
        return F.normalize(x, p=2, dim=1)
```

#### 4.2.3 Multi-Stage Retrieval

```python
class EnhancedRetriever:
    """
    Multi-stage retrieval pipeline with re-ranking and KG filtering.
    """
    def __init__(self, config):
        self.vector_store = VectorStore(config.vector_store)
        self.reranker = Reranker(config.reranker)
        self.kg_query = KnowledgeGraphQuery(config.knowledge_graph)
        self.feedback_collector = FeedbackCollector(config.feedback)
        
    def retrieve(self, query_embedding, query_text, fault_codes, top_k=10):
        """
        Multi-stage retrieval pipeline.
        
        Returns:
            List[Dict]: Ranked repair guide recommendations
        """
        # Stage 1: Vector search
        candidates = self.vector_store.search(
            query_embedding,
            top_k=100,
            filter_dict={"fault_codes": fault_codes}  # Optional filtering
        )
        
        # Stage 2: Re-ranking
        reranked = self.reranker.rerank(
            query_text,
            [c['text'] for c in candidates[:50]],
            top_k=20
        )
        
        # Stage 3: KG path scoring
        kg_scores = {}
        for candidate in reranked:
            procedure_id = candidate['procedure_id']
            # Find paths from fault codes to procedure
            paths = self.kg_query.find_paths(
                source_node=f"fault_{fault_codes[0]}",
                target_type="repair_procedure",
                target_id=procedure_id
            )
            kg_scores[procedure_id] = self.kg_query.score_paths(paths)
        
        # Stage 4: Combined scoring
        final_scores = []
        for candidate in reranked:
            procedure_id = candidate['procedure_id']
            
            # Get feedback score (if available)
            feedback_score = self.feedback_collector.get_procedure_score(
                procedure_id
            ) or 0.5  # Default neutral
            
            # Combined score
            combined_score = (
                0.4 * candidate['similarity'] +
                0.3 * candidate['rerank_score'] +
                0.2 * kg_scores.get(procedure_id, 0.0) +
                0.1 * feedback_score
            )
            
            candidate['combined_score'] = combined_score
            final_scores.append(candidate)
        
        # Sort by combined score
        final_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_scores[:top_k]
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goals:**
- Set up project structure
- Implement basic embedding encoders
- Set up vector store (Qdrant)
- Create database integration layer

**Tasks:**
1. Create project directory structure
2. Install dependencies (requirements.txt)
3. Implement `FaultCodeEncoder` with E5-Mistral
4. Implement `OBDDataEncoder` with structured neural network
5. Implement `MultiModalEncoder` with cross-attention
6. Set up Qdrant vector store
7. Create database connection layer for BMW ISTA databases
8. Write unit tests for encoders

**Deliverables:**
- Working embedding system
- Vector store initialized
- Database integration complete

### Phase 2: Retrieval Pipeline (Weeks 3-4)

**Goals:**
- Implement multi-stage retrieval
- Integrate re-ranking
- Build knowledge graph query interface
- Create combined scoring system

**Tasks:**
1. Implement `VectorStore` interface
2. Implement `Reranker` with Cohere API or local model
3. Build knowledge graph from BMW databases
4. Implement `KnowledgeGraphQuery` interface
5. Implement `Ranker` with combined scoring
6. Create `EnhancedRetriever` orchestrator
7. Write integration tests

**Deliverables:**
- Complete retrieval pipeline
- Knowledge graph built
- Re-ranking integrated

### Phase 3: Conversational RAG (Weeks 5-6)

**Goals:**
- Implement LLM provider abstraction
- Build clarification question generation
- Create query expansion mechanism
- Implement session management

**Tasks:**
1. Implement `LLMProvider` abstract base class
2. Implement OpenAI and Anthropic clients
3. Create prompt templates for clarification
4. Implement `ConversationalRAG` orchestrator
5. Build ambiguity detection logic
6. Implement query expansion
7. Create session management system
8. Write tests for conversational flow

**Deliverables:**
- Conversational RAG system
- Clarification question generation
- Multi-turn conversation support

### Phase 4: Self-Improvement (Weeks 7-8)

**Goals:**
- Implement feedback collection system
- Build reward model
- Create embedding fine-tuning pipeline
- Implement active learning

**Tasks:**
1. Implement `FeedbackCollector` with SQLite storage
2. Build `FeedbackAnalyzer` for statistics
3. Implement `RewardModel` neural network
4. Create `EmbeddingTrainer` with contrastive learning
5. Implement `ActiveLearning` uncertainty sampling
6. Create training scripts
7. Write tests for feedback and training

**Deliverables:**
- Feedback collection system
- Fine-tuning pipeline
- Active learning module

### Phase 5: API & Integration (Weeks 9-10)

**Goals:**
- Build FastAPI server
- Create API endpoints
- Integrate with existing BMW ISTA scripts
- Write documentation

**Tasks:**
1. Implement FastAPI server
2. Create API endpoints (`/query`, `/clarify`, `/feedback/*`)
3. Integrate with `generate_repair_guide.py`
4. Create indexing scripts for repair guides
5. Write comprehensive documentation
6. Create example usage scripts
7. Performance testing and optimization

**Deliverables:**
- Production-ready API server
- Integration with existing codebase
- Complete documentation

### Phase 6: Testing & Refinement (Weeks 11-12)

**Goals:**
- Comprehensive testing
- Performance optimization
- Bug fixes
- User acceptance testing

**Tasks:**
1. End-to-end testing
2. Performance profiling and optimization
3. Bug fixes
4. User acceptance testing with real diagnostic cases
5. Documentation updates
6. Deployment preparation

**Deliverables:**
- Tested and optimized system
- Production deployment ready

---

## 6. Integration Points

### 6.1 BMW ISTA Database Integration

**Existing Database Tables:**
- `XEP_FAULTCODES`: Fault code definitions
- `XEP_FAULTLABELS`: Fault code descriptions
- `XEP_ECUVARIANTS`: ECU information
- `XEP_INFOOBJECTS`: Repair procedures
- `XEP_INFOSEGMENTS`: Procedure content
- `RG_ECUFAULT_DOCIDS`: Fault-repair mappings
- `XEP_REFDIAGOBJECTS`: Fault-diagnostic relationships

**Integration Strategy:**
1. Use existing `IstaDatabase` class from `generate_repair_guide.py`
2. Extend with methods for embedding-based retrieval
3. Maintain compatibility with existing rule-based filtering
4. Add new tables for feedback and embeddings metadata

### 6.2 Existing Scripts Integration

**`generate_repair_guide.py`:**
- Add option to use MIST mapping layer instead of keyword search
- Integrate clarification questions into workflow
- Use MIST recommendations for guide selection

**New Integration Points:**
```python
# In generate_repair_guide.py
from mist.src.retrieval.conversational_rag import ConversationalRAG

# Initialize MIST
mist_rag = ConversationalRAG(config_path="mist/config/retrieval_config.yaml")

# Use MIST for retrieval
response = mist_rag.query(
    fault_codes=[data['fault_code']],
    obd_data=obd_readings,  # From OBD port
    vehicle_context=vehicle_profile
)

# Handle clarification if needed
if response['needs_clarification']:
    questions = response['clarification_questions']
    # Present to user, collect responses
    refined_response = mist_rag.clarify(
        session_id=response['session_id'],
        responses=user_responses
    )
    recommendations = refined_response['recommendations']
else:
    recommendations = response['recommendations']
```

### 6.3 OBD Data Collection

**Integration with OBD-II Interface:**
- Use `python-OBD` library for OBD-II communication
- Standard PIDs (Parameter IDs) for sensor readings
- Real-time data collection during diagnosis

**OBD Parameters to Collect:**
- Engine RPM
- Vehicle speed
- Throttle position
- Coolant temperature
- Intake air temperature
- MAF (Mass Air Flow)
- Fuel pressure
- Intake manifold pressure
- Timing advance
- Fuel level
- Barometric pressure
- O2 sensor readings
- Catalyst temperature

---

## 7. Self-Improvement Mechanisms

### 7.1 Feedback Collection

**Feedback Types:**
1. **Explicit Ratings**: 1-5 scale for recommendation quality
2. **Repair Outcomes**: success/failure/partial
3. **Conversation Corrections**: User corrections to clarification questions
4. **Selected Guide Tracking**: Which guide was actually used

**Storage Schema:**
```sql
CREATE TABLE feedback_sessions (
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
```

### 7.2 Reward Model Training

**Architecture:**
```python
class RewardModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_embedding, doc_embedding):
        # Combine embeddings
        combined = torch.cat([query_embedding, doc_embedding], dim=1)
        if combined.size(1) > self.network[0].in_features:
            # Use difference if too large
            combined = query_embedding - doc_embedding
        
        return self.network(combined)
```

**Training Process:**
1. Collect feedback with ratings ≥ 3
2. Create positive pairs (query, relevant_doc) and negative pairs (query, irrelevant_doc)
3. Train reward model to predict feedback scores
4. Use reward model to score new recommendations

### 7.3 Embedding Fine-Tuning

**Contrastive Learning:**
```python
def contrastive_loss(anchor, positive, negatives, temperature=0.05):
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        anchor: Query embedding
        positive: Relevant document embedding
        negatives: List of irrelevant document embeddings
        temperature: Temperature parameter
    """
    # Positive pair similarity
    pos_sim = F.cosine_similarity(anchor, positive, dim=1) / temperature
    
    # Negative pair similarities
    neg_sims = []
    for neg in negatives:
        neg_sim = F.cosine_similarity(anchor, neg, dim=1) / temperature
        neg_sims.append(neg_sim)
    
    # Combine
    all_sims = torch.cat([pos_sim.unsqueeze(1)] + [n.unsqueeze(1) for n in neg_sims], dim=1)
    
    # InfoNCE loss
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(all_sims, labels)
    
    return loss
```

**Fine-Tuning Pipeline:**
1. Load pre-trained encoders
2. Create `FeedbackDataset` from collected feedback
3. Train with contrastive loss
4. Validate on held-out feedback
5. Save checkpoints after each epoch
6. Re-index vector store with updated embeddings

### 7.4 Active Learning

**Uncertainty Sampling:**
```python
def identify_uncertain_cases(candidates, threshold=0.65):
    """
    Identify cases that need clarification or human review.
    
    Criteria:
    1. Top score < threshold
    2. Score variance < 0.02 (very similar scores)
    3. Missing critical OBD parameters
    """
    uncertain = []
    
    for candidate_set in candidates:
        scores = [c['combined_score'] for c in candidate_set]
        
        # Check top score
        if max(scores) < threshold:
            uncertain.append(candidate_set)
            continue
        
        # Check variance
        if len(scores) >= 3:
            top3_scores = sorted(scores, reverse=True)[:3]
            variance = np.var(top3_scores)
            if variance < 0.02:
                uncertain.append(candidate_set)
                continue
    
    return uncertain
```

---

## 8. Database Schema Extensions

### 8.1 New Tables for MIST

**`mist_embeddings`:**
```sql
CREATE TABLE mist_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id TEXT NOT NULL,
    embedding BLOB,  -- 768-dim vector (numpy array)
    embedding_version INTEGER,  -- Version for fine-tuning
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (procedure_id) REFERENCES XEP_INFOOBJECTS(ID)
);

CREATE INDEX idx_mist_embeddings_procedure ON mist_embeddings(procedure_id);
CREATE INDEX idx_mist_embeddings_version ON mist_embeddings(embedding_version);
```

**`mist_feedback`:**
```sql
CREATE TABLE mist_feedback (
    feedback_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    procedure_id TEXT,
    rating INTEGER,  -- 1-5
    repair_outcome TEXT,  -- success/failure/partial
    feedback_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
);

CREATE INDEX idx_mist_feedback_session ON mist_feedback(session_id);
CREATE INDEX idx_mist_feedback_procedure ON mist_feedback(procedure_id);
```

**`mist_training_checkpoints`:**
```sql
CREATE TABLE mist_training_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    epoch INTEGER,
    loss REAL,
    validation_loss REAL,
    embedding_version INTEGER,
    checkpoint_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 8.2 Extended Knowledge Graph

**Additional Node Types:**
- `symptom`: User-reported symptoms
- `vehicle_model`: Vehicle model/chassis information
- `component`: Vehicle components (e.g., "water pump", "ignition coil")

**Additional Edge Types:**
- `has_symptom`: `fault_code → symptom`
- `affects_component`: `fault_code → component`
- `applies_to_model`: `repair_procedure → vehicle_model`

---

## 9. Configuration Files

### 9.1 Embedding Configuration

```yaml
# config/embedding_config.yaml
models:
  fault_code:
    model_name: intfloat/e5-mistral-7b-instruct
    projection_dim: 768
    max_length: 512
    device: auto  # auto, cpu, cuda
  
  obd_data:
    input_dim: 128
    hidden_dim: 256
    output_dim: 768
    attention_heads: 8
  
  fusion:
    type: cross_attention
    hidden_dim: 768
    num_heads: 8
    dropout: 0.1
  
  output_dimension: 768

training:
  batch_size: 32
  learning_rate: 1e-5
  num_epochs: 10
  warmup_steps: 100
  weight_decay: 0.01
  temperature: 0.05  # For contrastive loss
```

### 9.2 Retrieval Configuration

```yaml
# config/retrieval_config.yaml
vector_store:
  provider: qdrant
  collection_name: repair_guides_enhanced
  distance_metric: cosine
  vector_size: 768
  url: http://localhost:6333  # Or local file path

retrieval:
  initial_k: 100
  rerank_k: 50
  final_k: 10
  min_similarity: 0.5

reranking:
  enabled: true
  provider: cohere  # cohere or local
  model: rerank-english-v3.0  # Cohere model
  # Or for local:
  # model: cross-encoder/ms-marco-MiniLM-L-12-v2
  top_k: 50

ranking:
  embedding_similarity: 0.4
  rerank_score: 0.3
  kg_path_score: 0.2
  feedback_score: 0.1

knowledge_graph:
  enabled: true
  graph_path: data/knowledge_graph.graphml
  max_path_length: 3
  min_path_score: 0.3

clarification:
  enabled: true
  ambiguity_threshold: 0.65
  score_variance_threshold: 0.02
  max_questions: 3
  max_clarifications_per_session: 3
```

### 9.3 LLM Configuration

```yaml
# config/llm_config.yaml
providers:
  primary: openai
  fallback:
    - anthropic
    - open_source

openai:
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
  temperature: 0.7
  max_tokens: 1000
  timeout: 30

anthropic:
  model: claude-3-5-sonnet-20241022
  api_key_env: ANTHROPIC_API_KEY
  temperature: 0.7
  max_tokens: 1000
  timeout: 30

open_source:
  provider: ollama
  model: llama3.1:70b
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 1000

prompts:
  clarification:
    system: |
      You are a diagnostic assistant helping to clarify automotive fault diagnosis.
      Analyze the fault codes and OBD data, then ask 1-3 clarifying questions
      to help narrow down the diagnosis.
      
      Focus on:
      - Missing critical information
      - Ambiguous symptoms
      - Vehicle-specific context
      
      Return only the questions, numbered or bulleted.
    
    user_template: |
      Fault Codes: {fault_codes}
      OBD Data: {obd_data}
      Top Recommendations: {top_candidates}
      
      Generate clarifying questions to improve diagnosis accuracy.
  
  query_expansion:
    system: |
      You are a query expansion assistant for automotive diagnostics.
      Expand the original query with context from user responses.
    
    user_template: |
      Original Query: {original_query}
      User Responses: {user_responses}
      
      Expand the query with relevant context.
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Embedding Tests:**
- Test fault code encoder with various fault descriptions
- Test OBD encoder with different parameter combinations
- Test multimodal fusion with edge cases
- Verify embedding dimensions and normalization

**Retrieval Tests:**
- Test vector store search
- Test re-ranking with known query-document pairs
- Test knowledge graph path finding
- Test combined scoring logic

**Conversational RAG Tests:**
- Test ambiguity detection logic
- Test clarification question generation
- Test query expansion
- Test session management

### 10.2 Integration Tests

**End-to-End Flow:**
1. Input: Fault codes + OBD data
2. Process through embedding → retrieval → ranking
3. Verify recommendations are relevant
4. Test clarification flow
5. Test feedback collection

**Database Integration:**
- Test BMW ISTA database queries
- Test knowledge graph construction
- Test vector store indexing

### 10.3 Performance Tests

**Latency:**
- Embedding generation time
- Vector search time
- Re-ranking time
- End-to-end query time

**Throughput:**
- Queries per second
- Concurrent request handling

**Scalability:**
- Vector store size limits
- Knowledge graph size limits
- Database query performance

---

## 11. Deployment Considerations

### 11.1 Production Setup

**Infrastructure:**
- Qdrant server (not local file storage)
- GPU for embedding generation (optional but recommended)
- Sufficient RAM for model loading
- Fast storage for vector store

**Monitoring:**
- API request/response logging
- Embedding generation time
- Retrieval performance metrics
- Feedback collection rates
- Model training progress

**Scaling:**
- Horizontal scaling for API server
- Qdrant cluster for vector store
- Load balancing for multiple instances

### 11.2 Security

**API Security:**
- Authentication/authorization
- Rate limiting
- Input validation
- SQL injection prevention

**Data Privacy:**
- Anonymize feedback data
- Secure storage of embeddings
- Access control for databases

---

## 12. Future Enhancements

### 12.1 Advanced Features

**Multi-Language Support:**
- Use multilingual embedding models (BGE-M3)
- Support non-English fault descriptions
- Translate repair guides

**Temporal Analysis:**
- Track OBD data over time
- Detect patterns in sensor readings
- Predict fault progression

**Explainability:**
- Visualize knowledge graph paths
- Show why recommendations were made
- Highlight important OBD parameters

### 12.2 Research Directions

**Graph Neural Networks:**
- Learn node embeddings from graph structure
- Improve relationship understanding
- Better path scoring

**Few-Shot Learning:**
- Adapt to new fault codes quickly
- Learn from minimal examples
- Transfer learning across vehicle models

**Reinforcement Learning:**
- Optimize clarification questions
- Learn optimal retrieval strategies
- Adaptive ranking weights

---

## 13. Success Metrics

### 13.1 Accuracy Metrics

**Retrieval Accuracy:**
- Precision@K (K=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

**Clarification Effectiveness:**
- Reduction in ambiguity after clarification
- User satisfaction with questions
- Improvement in recommendation quality

### 13.2 User Experience Metrics

**Response Time:**
- Average query time
- Clarification response time
- 95th percentile latency

**User Satisfaction:**
- Average rating
- Repair success rate
- User retention

### 13.3 Learning Metrics

**Model Improvement:**
- Reduction in loss over time
- Improvement in retrieval accuracy
- Feedback incorporation rate

**Active Learning:**
- Cases identified for review
- Human feedback collection rate
- Model improvement from feedback

---

## 14. Conclusion

This implementation plan provides a comprehensive roadmap for building an enhanced AI-driven mapping layer for automotive diagnostics. By combining cutting-edge embedding models, multi-stage retrieval, conversational RAG, and self-improvement mechanisms, we can create a system that continuously learns and improves from real-world diagnostic experiences.

The system builds upon the existing BMW ISTA reverse engineering kit while adding intelligent capabilities that make diagnostic recommendations more accurate, explainable, and user-friendly.

**Next Steps:**
1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Regular progress reviews and adjustments

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** AI System Design  
**Status:** Draft - Ready for Review

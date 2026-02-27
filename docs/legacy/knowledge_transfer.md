# MIST Knowledge Transfer Document

## Document Purpose

This document provides comprehensive technical specifications for the MIST (Multi-modal Intelligent Service Technician) system. It is designed for AI-driven code generation to recreate the system with enhanced database mapping capabilities. The document covers architecture, implementation details, design patterns, and all component specifications.

---

## 1. Project Overview

### 1.1 Purpose and Goals

MIST is an AI-powered automotive diagnostic system that:
- Maps fault codes and OBD (On-Board Diagnostics) data to repair guide recommendations
- Uses multi-modal embeddings to combine text (fault codes) and structured data (OBD sensors)
- Provides conversational clarification for ambiguous diagnostic cases
- Continuously self-improves through user feedback and repair outcomes
- Leverages knowledge graphs from BMW diagnostic databases for explainable recommendations

### 1.2 Core Problem Statement

Automotive diagnostics require mapping:
- **Fault codes** (text descriptions of problems)
- **OBD sensor data** (structured real-time vehicle parameters)
- **Repair procedures** (step-by-step guides)

The challenge is creating a unified representation that captures semantic relationships between these heterogeneous data types while maintaining explainability and learning from user feedback.

### 1.3 Key Capabilities and Features

1. **Multi-Modal Understanding**: Combines fault code text and OBD sensor data in unified 768-dimensional embedding space
2. **Conversational RAG**: LLM-guided clarification questions for precise retrieval
3. **Self-Improving**: Learns from user feedback and repair outcomes via RLHF
4. **Knowledge Graph Integration**: Leverages BMW diagnostic database relationships
5. **Multi-Stage Retrieval**: Combines vector search, re-ranking, and KG constraints
6. **Provider-Agnostic LLM**: Supports OpenAI, Anthropic, and open-source models with fallback chain

### 1.4 Domain Context

**Automotive Diagnostics Domain:**
- **Fault Codes**: Standardized codes (e.g., P0300 - Random/Multiple Cylinder Misfire Detected)
- **OBD Data**: Real-time sensor readings (RPM, temperature, pressure, etc.)
- **ECUs**: Electronic Control Units managing different vehicle systems
- **Repair Procedures**: Step-by-step diagnostic and repair instructions
- **Diagnostic Trees**: Hierarchical diagnostic workflows

**BMW Diagnostic Database Structure:**
- `XEP_FAULTCODES`: Fault code definitions
- `XEP_ECUVARIANTS`: ECU information
- `XEP_INFOOBJECTS`: Repair procedures and information objects
- `XEP_DIAGNOSISOBJECTS`: Diagnostic test procedures
- `RG_ECUFAULT_DOCIDS`: Mapping between fault codes and repair documents
- `XEP_REFDIAGOBJECTS`: Relationships between fault codes and diagnostic objects

---

## 2. Architecture

### 2.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server (FastAPI)                    │
│  /query | /clarify | /feedback/* | /health                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ConversationalRAG (Orchestrator)               │
│  - Query Processing                                             │
│  - Clarification Generation                                     │
│  - Multi-turn Conversation Management                           │
└──────┬──────────────────────┬──────────────────────┬────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Multi-Modal  │    │   Vector Store   │    │  Knowledge Graph │
│   Encoder    │    │    (ChromaDB)      │    │   (NetworkX)     │
│              │    │                  │    │                  │
│ FaultCode +  │───▶│ Repair Guides    │    │ Fault→ECU→      │
│ OBD Data     │    │ Embeddings       │    │ Diagnostic→      │
│              │    │                  │    │ Repair Paths     │
└──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘
       │                     │                        │
       │                     ▼                        │
       │            ┌──────────────────┐              │
       │            │     Ranker       │◀─────────────┘
       │            │                  │
       │            │ Multi-stage:     │
       │            │ - Embedding      │
       │            │ - KG Path Score  │
       │            │ - Feedback Score │
       │            │ - Recency        │
       └────────────┼──────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Provider Abstraction                      │
│  OpenAI | Anthropic | Open Source (Ollama)                      │
│  Fallback Chain: Primary → Fallback[0] → Fallback[1]            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagrams

#### 2.2.1 Query Processing Flow

```
User Input (Fault Codes + OBD Data)
    │
    ▼
[Multi-Modal Encoder]
    │
    ├─→ FaultCodeEncoder (sentence-transformers) → 384-dim
    └─→ OBDDataEncoder (neural network) → 384-dim
    │
    ▼
[Cross-Attention Fusion]
    │
    ├─→ Fault attends to OBD
    └─→ OBD attends to Fault
    │
    ▼
[Fusion Layer] → 768-dim unified embedding
    │
    ▼
[Vector Store Search] (ChromaDB, top-K=50)
    │
    ▼
[Re-ranking] (Cross-encoder, optional)
    │
    ▼
[Ranker] (Combined scoring)
    │
    ├─→ Embedding similarity (40%)
    ├─→ KG path score (30%)
    ├─→ Feedback score (20%)
    └─→ Recency score (10%)
    │
    ▼
[Ambiguity Detection]
    │
    ├─→ Low confidence? → Generate clarification questions
    └─→ High confidence? → Return recommendations
```

#### 2.2.2 Conversational Clarification Flow

```
Initial Query
    │
    ▼
[Process Query] → Low confidence detected
    │
    ▼
[LLM Clarification Generation]
    │
    ├─→ System Prompt: "You are a diagnostic assistant..."
    └─→ User Prompt: Fault codes + OBD data + Top candidates
    │
    ▼
[Parse Questions] (1-3 questions)
    │
    ▼
Return to User: {recommendations, clarification_questions}
    │
    ▼
User Responses
    │
    ▼
[Query Expander] (LLM-based)
    │
    ├─→ Original query + User responses
    └─→ Expanded query with context
    │
    ▼
[Re-process Query] with expanded context
    │
    ▼
Refined Recommendations
```

#### 2.2.3 Self-Improvement Loop

```
User Feedback
    │
    ├─→ Explicit Rating (1-5)
    ├─→ Repair Outcome (success/failure/partial)
    └─→ Conversation Corrections
    │
    ▼
[FeedbackCollector] (SQLite storage)
    │
    ▼
[FeedbackAnalyzer]
    │
    ├─→ Statistics
    ├─→ Low-confidence case identification
    ├─→ Positive/negative example extraction
    └─→ Training data preparation
    │
    ▼
[EmbeddingTrainer]
    │
    ├─→ Contrastive Learning
    ├─→ Reward Model (RLHF)
    └─→ Fine-tuning MultiModalEncoder
    │
    ▼
Updated Embeddings → Improved Retrieval
```

### 2.3 Component Interaction Patterns

**Dependency Injection Pattern:**
- Components receive configuration paths or use centralized `get_paths()`
- Components are initialized with default configs but can be overridden
- Factory pattern for LLM providers

**Lazy Initialization:**
- Knowledge graph loaded only if enabled in config
- Re-ranking model loaded only if enabled
- Components initialized on first use

**Session Management:**
- Session IDs track multi-turn conversations
- FeedbackCollector maintains session state
- ConversationalRAG uses session context for clarification

### 2.4 Modular Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Interface Abstraction**: LLM providers use abstract base class
3. **Configuration-Driven**: YAML configs control behavior
4. **Path Centralization**: Single source of truth for file paths
5. **Error Handling**: Try-except blocks with graceful degradation

### 2.5 Integration Points

**Between Components:**
- `ConversationalRAG` → `MultiModalEncoder`, `VectorStore`, `Ranker`, `LLMProvider`
- `Ranker` → `KnowledgeGraphQuery`, `FeedbackCollector` (optional)
- `VectorStore` → `MultiModalEncoder` (for indexing)
- `KnowledgeGraphBuilder` → SQLite database → NetworkX graph

**External Dependencies:**
- ChromaDB (vector database)
- SQLite (BMW diagnostic databases, feedback storage)
- OpenAI/Anthropic APIs (LLM providers)
- NetworkX (knowledge graph)
- PyTorch (embeddings, training)

---

## 3. Component Specifications

### 3.1 Multi-Modal Embeddings

#### 3.1.1 Architecture Overview

The multi-modal embedding system combines:
- **FaultCodeEncoder**: Text encoder for fault code descriptions
- **OBDDataEncoder**: Structured encoder for OBD JSON data
- **MultiModalEncoder**: Cross-attention fusion layer

**Dimension Flow:**
```
Fault Code Text → 384-dim (FaultCodeEncoder)
OBD Data JSON → 384-dim (OBDDataEncoder)
    │
    ├─→ Projection to 768-dim (hidden_dim)
    ├─→ Cross-Attention Fusion
    └─→ Final Output: 768-dim
```

#### 3.1.2 FaultCodeEncoder

**Location:** `src/embeddings/fault_code_encoder.py`

**Technical Approach:**
- Uses `sentence-transformers` library
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: 384
- Max sequence length: 512 tokens
- Normalizes embeddings (L2 normalization)

**Input Specifications:**
- Single string or list of strings
- Fault code descriptions (e.g., "Random/Multiple Cylinder Misfire Detected")

**Output Specifications:**
- PyTorch Tensor: `(batch_size, 384)` or `(384,)` for single input
- Normalized embeddings (if `normalize=True`)

**Configuration:**
```yaml
models:
  fault_code:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    max_length: 512
    device: auto  # auto, cpu, cuda
```

**Key Methods:**
- `encode(texts, normalize=True) -> torch.Tensor`
- `get_dimension() -> int`

#### 3.1.3 OBDDataEncoder

**Location:** `src/embeddings/obd_data_encoder.py`

**Technical Approach:**
- PyTorch neural network module (`nn.Module`)
- Normalizes OBD parameters to fixed-size feature vector (128 features)
- Feature extraction: `128 → 256 → 512 → 384`
- Handles variable-length OBD data by padding/truncating

**OBD Parameter Normalization:**
- Common PIDs with known ranges:
  - `engine_rpm`: (0, 8000)
  - `vehicle_speed`: (0, 255)
  - `throttle_position`: (0, 100)
  - `coolant_temp`: (-40, 215)
  - `intake_temp`: (-40, 215)
  - `maf_airflow`: (0, 655.35)
  - `fuel_pressure`: (0, 765)
  - `intake_pressure`: (0, 255)
  - `timing_advance`: (-64, 63.5)
  - `fuel_level`: (0, 100)
  - `barometric_pressure`: (0, 255)
- Missing values: Set to 0.0
- Additional numeric values: Normalized by max absolute value, limited to 10 features

**Input Specifications:**
- Dictionary: `{param_name: value, ...}`
- List of dictionaries for batch processing
- Values can be int, float, or missing

**Output Specifications:**
- PyTorch Tensor: `(batch_size, 384)` or `(384,)` for single input
- Normalized embeddings (if configured)

**Configuration:**
```yaml
models:
  obd_data:
    type: structured_json
    normalization: true
    dimension: 384
```

**Key Methods:**
- `forward(obd_data) -> torch.Tensor`
- `encode(obd_data) -> torch.Tensor` (eval mode)
- `normalize_obd_data(obd_data) -> torch.Tensor`
- `get_dimension() -> int`

#### 3.1.4 MultiModalEncoder

**Location:** `src/embeddings/multimodal_encoder.py`

**Technical Approach: Cross-Attention Fusion**

**Architecture:**
1. **Component Encoders:**
   - `FaultCodeEncoder`: 384-dim
   - `OBDDataEncoder`: 384-dim

2. **Projection Layers:**
   - `fault_projection`: `Linear(384 → 768)`
   - `obd_projection`: `Linear(384 → 768)`

3. **Cross-Attention Mechanism:**
   - `MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)`
   - Bidirectional attention:
     - Fault codes attend to OBD data
     - OBD data attends to fault codes

4. **Fusion Layer:**
   - Input: Concatenated attended embeddings `(768 * 2 = 1536)`
   - Architecture: `1536 → 768 → ReLU → Dropout(0.1) → 768`
   - Output: 768-dim unified embedding

5. **Layer Normalization:**
   - Applied after cross-attention

6. **Final Normalization:**
   - L2 normalization on output

**Input Specifications:**
- `fault_codes`: String or list of strings
- `obd_data`: Dictionary or list of dictionaries

**Output Specifications:**
- PyTorch Tensor: `(batch_size, 768)` or `(768,)` for single input
- Normalized embeddings

**Configuration:**
```yaml
models:
  fusion:
    type: cross_attention
    hidden_dim: 768
    num_heads: 8
    dropout: 0.1
  output_dimension: 768
```

**Key Methods:**
- `forward(fault_codes, obd_data) -> torch.Tensor`
- `encode(fault_codes, obd_data) -> torch.Tensor` (eval mode)
- `get_dimension() -> int`

**Batch Handling:**
- If batch sizes differ, repeats single-item batches to match
- Adds sequence dimension `(batch, seq_len=1, hidden_dim)` for attention
- Removes sequence dimension after attention

### 3.2 Vector Store

#### 3.2.1 Provider: ChromaDB

**Location:** `src/retrieval/vector_store.py`

**Technical Approach:**
- Uses ChromaDB client library
- Local file-based storage (development)
- Can be upgraded to ChromaDB server for production

**Collection Configuration:**
- Collection name: `repair_guides` (configurable)
- Distance metric: Cosine similarity
- Vector size: 768 (matches MultiModalEncoder output)

**Indexing Strategy:**
- Batch insertion (default batch_size=100)
- Upsert operations (updates if document ID exists)
- Metadata stored in payload:
  - `text`: Procedure text content
  - `title`: Procedure title
  - `procedure_id`: Unique procedure identifier
  - `procedure_name`: Procedure name
  - `fault_codes`: List of associated fault codes
  - `ecu_category`: ECU category
  - `metadata`: Additional metadata dict

**Search Mechanisms:**
- Cosine similarity search
- Top-K retrieval (configurable)
- Optional metadata filtering:
  - Field conditions (exact match)
  - Can filter by `procedure_id`, `fault_codes`, `ecu_category`, etc.

**Configuration:**
```yaml
vector_store:
  provider: chromadb
  collection_name: repair_guides
  distance_metric: cosine
  vector_size: 768
```

**Key Methods:**
- `add_documents(embeddings, documents, batch_size=100)`
- `search(query_embedding, top_k=10, filter_dict=None) -> List[Dict]`
- `update_document(doc_id, embedding=None, payload=None)`
- `delete_document(doc_id)`
- `get_collection_info() -> Dict`

**Search Result Format:**
```python
{
    'id': str,  # Document ID
    'score': float,  # Similarity score (0-1)
    'text': str,
    'title': str,
    'procedure_id': str,
    'procedure_name': str,
    'fault_codes': List[str],
    'ecu_category': str,
    'metadata': Dict
}
```

### 3.3 LLM Provider Abstraction

#### 3.3.1 Provider Pattern Implementation

**Location:** `src/llm/provider.py`

**Abstract Base Class:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(messages: List[Dict[str, str]], **kwargs) -> str
    
    @abstractmethod
    def generate_stream(messages: List[Dict[str, str]], **kwargs) -> Iterator[str]
```

**Factory Pattern:**
- `LLMProviderFactory.create_provider(config_path) -> LLMProvider`
- Tries providers in order: Primary → Fallback[0] → Fallback[1]
- Raises error if all providers fail

#### 3.3.2 Supported Providers

**1. OpenAI Client**
- **Location:** `src/llm/openai_client.py`
- **Model:** `gpt-4o` (configurable)
- **API:** OpenAI Chat Completions API
- **Environment Variable:** `OPENAI_API_KEY`
- **Features:** Non-streaming and streaming generation
- **Configuration:**
  ```yaml
  openai:
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    temperature: 0.7
    max_tokens: 1000
    timeout: 30
  ```

**2. Anthropic Client**
- **Location:** `src/llm/anthropic_client.py`
- **Model:** `claude-3-5-sonnet-20241022` (configurable)
- **API:** Anthropic Messages API
- **Environment Variable:** `ANTHROPIC_API_KEY`
- **Features:** System message handling, streaming support
- **Message Conversion:** Converts standard format to Anthropic format (system/user/assistant)
- **Configuration:**
  ```yaml
  anthropic:
    model: claude-3-5-sonnet-20241022
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.7
    max_tokens: 1000
    timeout: 30
  ```

**3. Open Source Client**
- **Location:** `src/llm/open_source_client.py`
- **Provider:** Ollama (default, configurable)
- **Model:** `llama3.1:70b` (configurable)
- **Base URL:** `http://localhost:11434` (configurable)
- **Library:** LangChain (`ChatOllama`)
- **Features:** Local/self-hosted LLM support
- **Configuration:**
  ```yaml
  open_source:
    provider: ollama
    model: llama3.1:70b
    base_url: http://localhost:11434
    temperature: 0.7
    max_tokens: 1000
  ```

#### 3.3.3 Fallback Chain Mechanism

**Configuration:**
```yaml
providers:
  primary: openai
  fallback:
    - anthropic
    - open_source
```

**Behavior:**
1. Try to initialize primary provider
2. If fails, try fallback[0]
3. If fails, try fallback[1]
4. If all fail, raise `RuntimeError`

**Error Handling:**
- Catches exceptions during initialization
- Logs error and continues to next provider
- Only raises error if all providers fail

#### 3.3.4 Prompt Template System

**Location:** `src/llm/prompt_templates.py`

**Template Types:**

**1. Clarification Prompt:**
- **System Prompt:** Defines role as diagnostic assistant
- **User Template:** Includes fault codes, OBD data, and context
- **Purpose:** Generate 1-3 clarifying questions

**Configuration:**
```yaml
prompts:
  clarification:
    system: |
      You are a diagnostic assistant helping to clarify automotive fault diagnosis.
      Analyze the fault codes and OBD data, then ask 1-3 clarifying questions...
    user_template: |
      Fault Codes: {fault_codes}
      OBD Data: {obd_data}
      Current Context: {context}
      
      Generate clarifying questions...
```

**2. Query Expansion Prompt:**
- **System Prompt:** Defines role as query expansion assistant
- **User Template:** Includes original query and user responses
- **Purpose:** Expand query with context from user responses

**Configuration:**
```yaml
prompts:
  query_expansion:
    system: |
      You are a query expansion assistant...
    user_template: |
      Original Query: {original_query}
      User Responses: {user_responses}
      Expand the query...
```

**Key Methods:**
- `get_clarification_prompt(fault_codes, obd_data, context) -> Dict[str, str]`
- `get_query_expansion_prompt(original_query, user_responses) -> Dict[str, str]`

### 3.4 Knowledge Graph

#### 3.4.1 Graph Structure (NetworkX)

**Location:** `src/knowledge/graph_builder.py`, `src/knowledge/graph_query.py`

**Graph Type:** `nx.MultiDiGraph` (Multi-directed graph, allows multiple edges)

**Node Types:**
1. **fault_code**: Fault code nodes
   - Attributes: `id`, `code`, `description`, `label`
   - Node ID format: `fault_{fault_id}`

2. **ecu**: ECU (Electronic Control Unit) nodes
   - Attributes: `id`, `name`, `title`, `label`
   - Node ID format: `ecu_{ecu_id}`

3. **repair_procedure**: Repair procedure nodes
   - Attributes: `id`, `name`, `title`, `label`
   - Node ID format: `procedure_{procedure_id}`

4. **diagnostic_object**: Diagnostic test procedure nodes
   - Attributes: `id`, `control_id`, `name`, `title`, `label`
   - Node ID format: `diag_{control_id}` or `diag_{diag_id}`

**Edge Types (Relationships):**
1. **affects_ecu**: `fault_code → ecu`
   - Weight: 1.0
   - Source: `XEP_FAULTCODES.ECUVARIANTID`

2. **has_diagnostic**: `fault_code → diagnostic_object`
   - Weight: `1.0 / priority` (higher priority = higher weight)
   - Source: `XEP_REFDIAGOBJECTS`

3. **has_repair**: `fault_code → repair_procedure`
   - Weight: 1.0
   - Source: `RG_ECUFAULT_DOCIDS`

4. **diagnostic_step**: `diagnostic_object → diagnostic_object` (parent-child)
   - Weight: 1.0
   - Source: `XEP_REFDIAGNOSISTREE`

#### 3.4.2 Relationship Types

**Fault → ECU → Diagnostic → Repair Path:**
```
fault_code --[affects_ecu]--> ecu
fault_code --[has_diagnostic]--> diagnostic_object
fault_code --[has_repair]--> repair_procedure
diagnostic_object --[diagnostic_step]--> diagnostic_object
```

**Path Scoring:**
- Sum of edge weights along path
- Normalized by max path length (typically 3)
- Higher score = stronger relationship

#### 3.4.3 Query Patterns

**Location:** `src/knowledge/graph_query.py`

**Key Query Methods:**

1. **find_paths(source_node, target_type, max_length=3) -> List[List[str]]**
   - Finds shortest paths from source to nodes of target type
   - Uses NetworkX `shortest_path` with weight consideration
   - Returns top 10 shortest paths

2. **get_neighbors(node_id, relationship_type=None) -> List[Dict]**
   - Gets neighboring nodes
   - Optional filter by relationship type
   - Returns node data with relationship info

3. **score_path(path) -> float**
   - Scores path based on edge weights
   - Returns total weight sum

4. **get_node_by_code(fault_code) -> Optional[str]**
   - Finds fault code node by code string
   - Returns node ID or None

5. **get_procedures_for_fault(fault_code) -> List[Dict]**
   - Gets repair procedures for a fault code
   - Returns list with path scores and procedure info

**Graph Storage:**
- Format: GraphML (NetworkX compatible)
- Location: `data/knowledge_graph.graphml`
- Loaded on `KnowledgeGraphQuery` initialization

**Database Extraction (KnowledgeGraphBuilder):**

**Tables Used:**
- `XEP_FAULTCODES`: Fault code definitions
- `XEP_FAULTLABELS`: Fault code descriptions
- `XEP_ECUVARIANTS`: ECU information
- `XEP_INFOOBJECTS`: Repair procedures
- `XEP_DIAGNOSISOBJECTS`: Diagnostic objects
- `XEP_REFDIAGOBJECTS`: Fault-diagnostic relationships
- `RG_ECUFAULT_DOCIDS`: Fault-repair mappings
- `XEP_REFDIAGNOSISTREE`: Diagnostic tree relationships

**Limits:**
- Fault codes: 10,000
- Repair procedures: 5,000
- Diagnostic objects: 5,000
- Relationships: 10,000 per type

### 3.5 Retrieval & Ranking

#### 3.5.1 Multi-Stage Retrieval Pipeline

**Location:** `src/retrieval/conversational_rag.py`, `src/retrieval/ranker.py`

**Pipeline Stages:**

**Stage 1: Initial Vector Search**
- Query embedding → Vector store search
- Retrieves top-K candidates (default K=50)
- Uses cosine similarity

**Stage 2: Re-ranking (Optional)**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-ranks top candidates (default top-10)
- Computes query-document similarity scores

**Stage 3: Combined Scoring**
- Weighted combination of multiple signals:
  - Embedding similarity: 40%
  - KG path score: 30%
  - Feedback score: 20%
  - Recency score: 10%

**Stage 4: Final Ranking**
- Sort by combined score (descending)
- Return top-N recommendations (default N=10)

#### 3.5.2 Ranking Algorithm

**Location:** `src/retrieval/ranker.py`

**Combined Score Calculation:**
```python
score = (
    embedding_weight * embedding_similarity +
    kg_weight * kg_path_score +
    feedback_weight * feedback_score +
    recency_weight * recency_score
)
```

**Default Weights:**
```yaml
ranking:
  embedding_similarity: 0.4
  kg_path_score: 0.3
  feedback_score: 0.2
  recency: 0.1
```

**Score Components:**

1. **Embedding Similarity:**
   - From vector store search result
   - Range: 0.0 to 1.0 (cosine similarity)

2. **KG Path Score:**
   - Finds paths from fault codes to repair procedures
   - Scores paths by edge weights
   - Normalizes by max path length
   - Range: 0.0 to 1.0

3. **Feedback Score:**
   - From FeedbackCollector (if available)
   - Default: 0.5 (neutral)
   - Range: 0.0 to 1.0

4. **Recency Score:**
   - From document metadata
   - Default: 0.5
   - Range: 0.0 to 1.0

#### 3.5.3 Re-ranking Strategy

**Cross-Encoder Re-ranking:**
- Model: `sentence-transformers` CrossEncoder
- Input: Query text + Document text pairs
- Output: Relevance scores
- Applied to top-K candidates before combined scoring

**Configuration:**
```yaml
reranking:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10
```

#### 3.5.4 Query Expansion Mechanism

**Location:** `src/retrieval/query_expander.py`

**Purpose:** Expand query with context from user clarification responses

**Process:**
1. Takes original query and user responses
2. Uses LLM to generate expanded query
3. Combines original and expanded query
4. Used in re-processing after clarification

**LLM Prompt:**
- System: Query expansion assistant role
- User: Original query + User responses
- Output: Expanded query text

**Configuration:**
```yaml
query_expansion:
  enabled: true
  max_expansions: 3
  expansion_weight: 0.3
```

### 3.6 Conversational RAG

#### 3.6.1 Multi-Turn Conversation Flow

**Location:** `src/retrieval/conversational_rag.py`

**Flow:**
```
1. Initial Query
   ├─→ Encode fault codes + OBD data
   ├─→ Vector search (top-K=50)
   ├─→ Rank candidates
   └─→ Check ambiguity

2. If Ambiguous:
   ├─→ Generate clarification questions (LLM)
   ├─→ Return recommendations + questions
   └─→ Wait for user responses

3. User Responses:
   ├─→ Expand query with responses
   ├─→ Re-process query
   └─→ Return refined recommendations
```

#### 3.6.2 Clarification Question Generation

**Process:**
1. Build context from top-3 candidates
2. Use LLM with clarification prompt template
3. Parse questions from LLM response (numbered list or bullet points)
4. Limit to 3 questions maximum

**Ambiguity Detection:**
- Top score < 0.6 → Needs clarification
- Top-3 scores have low variance (< 0.01) → Ambiguous
- Missing critical OBD parameters → Needs clarification

**Critical OBD Parameters:**
- `engine_rpm`
- `coolant_temp`

#### 3.6.3 Ambiguity Detection

**Method:** `_needs_clarification(candidates, fault_codes, obd_data) -> bool`

**Checks:**
1. Empty candidates → True
2. Top score < 0.6 → True
3. Top-3 score variance < 0.01 → True (very similar scores)
4. Missing critical OBD params → True

**Configuration:**
- Max clarifications per session: 3 (configurable)

#### 3.6.4 Session Management

**Session ID:**
- Generated by `FeedbackCollector.create_session()`
- UUID format
- Tracks multi-turn conversations

**Session State:**
- Stored in SQLite (`feedback_sessions` table)
- Includes: fault codes, OBD data, clarification questions, user responses, recommendations

**Session Retrieval:**
- `FeedbackCollector.get_session(session_id) -> Dict`
- Used to restore context for clarification responses

### 3.7 Feedback System

#### 3.7.1 Feedback Types

**Location:** `src/feedback/collector.py`

**1. Explicit Ratings:**
- Scale: 1-5 (1=worst, 5=best)
- Optional: Selected guide ID
- Stored in: `feedback_sessions.explicit_rating`

**2. Repair Outcomes:**
- Values: `success`, `failure`, `partial`
- Optional: Additional details dict
- Stored in: `feedback_sessions.repair_outcome`

**3. Conversation Corrections:**
- Dict with correction details
- Can be multiple per session
- Stored in: `feedback_sessions.conversation_corrections` (JSON array)

**4. Clarification Responses:**
- Questions and user responses
- Stored in: `feedback_sessions.clarification_questions`, `user_responses`

#### 3.7.2 Storage Mechanism (SQLite)

**Database Schema:**

**Table: `feedback_sessions`**
```sql
CREATE TABLE feedback_sessions (
    session_id TEXT PRIMARY KEY,
    fault_codes TEXT,  -- JSON array
    obd_data TEXT,  -- JSON object
    clarification_questions TEXT,  -- JSON array
    user_responses TEXT,  -- JSON array
    recommended_guides TEXT,  -- JSON array
    selected_guide TEXT,
    explicit_rating INTEGER,  -- 1-5
    repair_outcome TEXT,  -- success/failure/partial
    conversation_corrections TEXT,  -- JSON array
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

**Table: `feedback_events`**
```sql
CREATE TABLE feedback_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT,
    event_type TEXT,  -- explicit_rating, repair_outcome, conversation_correction
    event_data TEXT,  -- JSON object
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
)
```

**Database Location:**
- Path: `data/feedback/feedback.db`
- Created automatically if doesn't exist

#### 3.7.3 Analysis Capabilities

**Location:** `src/feedback/analyzer.py`

**Statistics:**
- Total sessions
- Rated sessions
- Average rating
- Repair outcomes breakdown
- Corrected sessions count
- Rating coverage percentage

**Low-Confidence Case Identification:**
- Cases with rating < 3
- Cases with repair outcome = 'failure'
- Cases with conversation corrections

**Training Data Extraction:**
- Positive examples: Rating >= 4 or outcome = 'success'
- Negative examples: Rating < 3 or outcome = 'failure'
- Minimum rating filter (default: 3)

**Key Methods:**
- `get_statistics() -> Dict`
- `identify_low_confidence_cases(threshold=0.6) -> List[Dict]`
- `get_positive_examples(min_rating=4) -> List[Dict]`
- `get_negative_examples() -> List[Dict]`

### 3.8 Self-Improvement

#### 3.8.1 Reward Model Architecture

**Location:** `src/learning/reward_model.py`

**Architecture:**
```python
RewardModel(
    input_dim=768,
    hidden_dim=512
)
```

**Network:**
```
Input (768) → Linear(768 → 512) → ReLU → Dropout(0.1) →
Linear(512 → 256) → ReLU → Dropout(0.1) →
Linear(256 → 1) → Sigmoid
```

**Output:** Single reward score (0.0 to 1.0)

**Reward Computation:**
- Combines query and candidate embeddings
- If combined > input_dim, uses difference instead
- Weighted combination: 70% feedback score + 30% predicted reward

#### 3.8.2 Embedding Fine-Tuning Pipeline

**Location:** `src/learning/embedding_trainer.py`

**Training Process:**

1. **Data Preparation:**
   - Load feedback from `FeedbackCollector`
   - Filter by minimum rating (default: 1)
   - Pre-compute embeddings for all feedback samples
   - Create labels: Rating normalized to [0,1] or outcome-based (success=1.0, failure=0.0)

2. **Dataset:**
   - `FeedbackDataset`: PyTorch Dataset
   - Stores pre-computed embeddings and labels
   - Returns `(embedding, label)` pairs

3. **Training Loop:**
   - Contrastive learning objective
   - Optimizer: AdamW (learning_rate=1e-5, weight_decay=0.01)
   - Batch size: 32 (configurable)
   - Epochs: 10 (configurable)

4. **Loss Function:**
   - Contrastive loss:
     - Positive pairs: High ratings (>0.7) → Maximize similarity
     - Negative pairs: Low ratings (<0.3) → Minimize similarity
   - Formula: `-log(similarity[positive]) + log(similarity[negative])`

5. **Checkpointing:**
   - Saves after each epoch
   - Location: `data/embeddings/checkpoints/checkpoint_epoch_{N}.pt`
   - Saves: Encoder state dict, Reward model state dict

**Configuration:**
```yaml
training:
  batch_size: 32
  learning_rate: 1e-5
  num_epochs: 10
  warmup_steps: 100
  weight_decay: 0.01

fine_tuning:
  enabled: true
  checkpoint_interval: 1000
  min_feedback_samples: 100
  validation_split: 0.2
```

**Minimum Requirements:**
- At least 10 feedback samples to start training

#### 3.8.3 Active Learning Strategy

**Location:** `src/learning/active_learning.py`

**Purpose:** Identify uncertain cases for human review

**Uncertainty Detection:**
1. Check if clarification needed (low confidence)
2. Check top recommendation score < threshold (default: 0.6)
3. Check score variance (top-3 scores very similar)

**Methods:**
- `identify_uncertain_cases(fault_codes, obd_data, threshold=0.6) -> bool`
- `get_uncertain_batch(cases, batch_size=10) -> List[Dict]`

**Use Case:**
- Prioritize cases for human review
- Improve training data quality
- Focus feedback collection on difficult cases

#### 3.8.4 Training Loop Design

**Training Script:** `scripts/train_embeddings.py`

**Process:**
1. Initialize `EmbeddingTrainer`
2. Check feedback statistics
3. Verify minimum samples (10)
4. Train embeddings
5. Save checkpoints

**Checkpoint Management:**
- Saved after each epoch
- Can load checkpoints for resuming training
- Format: PyTorch state dict

**Model Updates:**
- Fine-tuned encoder replaces original encoder
- Should be reloaded in production after training
- Vector store should be re-indexed with new embeddings

### 3.9 API Server

#### 3.9.1 Endpoint Specifications

**Location:** `src/api/server.py`

**Base URL:** `http://localhost:8000` (configurable)

**Endpoints:**

**1. POST `/query`**
- **Request:** `FaultCodeRequest`
  ```python
  {
      "fault_codes": List[str],
      "obd_data": Dict,
      "session_id": Optional[str]
  }
  ```
- **Response:** `QueryResponse`
  ```python
  {
      "recommendations": List[Recommendation],
      "needs_clarification": bool,
      "clarification_questions": Optional[List[str]],
      "session_id": str,
      "query_text": str
  }
  ```
- **Behavior:**
  - Creates session if `session_id` not provided
  - Processes query through ConversationalRAG
  - Returns recommendations and optional clarification questions

**2. POST `/clarify`**
- **Request:** `UserResponse`
  ```python
  {
      "session_id": str,
      "responses": List[str]
  }
  ```
- **Response:** `QueryResponse` (same as `/query`)
- **Behavior:**
  - Retrieves session data
  - Adds clarification responses
  - Re-processes query with expanded context
  - Returns refined recommendations

**3. POST `/feedback/rating`**
- **Request:** `RatingFeedback`
  ```python
  {
      "session_id": str,
      "rating": int,  # 1-5
      "selected_guide": Optional[str]
  }
  ```
- **Response:** `{"status": "success", "message": "Rating recorded"}`
- **Behavior:** Records explicit rating feedback

**4. POST `/feedback/outcome`**
- **Request:** `RepairOutcomeFeedback`
  ```python
  {
      "session_id": str,
      "outcome": str,  # success/failure/partial
      "details": Optional[Dict]
  }
  ```
- **Response:** `{"status": "success", "message": "Outcome recorded"}`
- **Behavior:** Records repair outcome feedback

**5. POST `/feedback/correction`**
- **Request:** `ConversationCorrection`
  ```python
  {
      "session_id": str,
      "correction": Dict
  }
  ```
- **Response:** `{"status": "success", "message": "Correction recorded"}`
- **Behavior:** Records conversation correction feedback

**6. GET `/feedback/statistics`**
- **Request:** None
- **Response:** `FeedbackStatistics`
  ```python
  {
      "total_sessions": int,
      "rated_sessions": int,
      "average_rating": float,
      "repair_outcomes": Dict[str, int],
      "corrected_sessions": int,
      "rating_coverage": float
  }
  ```
- **Behavior:** Returns feedback statistics

**7. GET `/health`**
- **Request:** None
- **Response:** `{"status": "healthy", "service": "MIST API"}`
- **Behavior:** Health check endpoint

#### 3.9.2 Request/Response Schemas

**Location:** `src/api/schemas.py`

**Pydantic Models:**

**FaultCodeRequest:**
```python
class FaultCodeRequest(BaseModel):
    fault_codes: List[str]
    obd_data: Dict
    session_id: Optional[str] = None
```

**Recommendation:**
```python
class Recommendation(BaseModel):
    id: str
    title: str
    procedure_name: str
    procedure_id: Optional[str]
    score: float
    text: Optional[str]
```

**QueryResponse:**
```python
class QueryResponse(BaseModel):
    recommendations: List[Recommendation]
    needs_clarification: bool
    clarification_questions: Optional[List[str]]
    session_id: str
    query_text: str
```

**RatingFeedback:**
```python
class RatingFeedback(BaseModel):
    session_id: str
    rating: int  # Field(ge=1, le=5)
    selected_guide: Optional[str]
```

**RepairOutcomeFeedback:**
```python
class RepairOutcomeFeedback(BaseModel):
    session_id: str
    outcome: str
    details: Optional[Dict]
```

**ConversationCorrection:**
```python
class ConversationCorrection(BaseModel):
    session_id: str
    correction: Dict
```

**FeedbackStatistics:**
```python
class FeedbackStatistics(BaseModel):
    total_sessions: int
    rated_sessions: int
    average_rating: float
    repair_outcomes: Dict[str, int]
    corrected_sessions: int
    rating_coverage: float
```

#### 3.9.3 Error Handling Patterns

**HTTP Exceptions:**
- `HTTPException(status_code=500, detail=str(e))` for server errors
- `HTTPException(status_code=404, detail="Session not found")` for missing sessions

**Try-Except Blocks:**
- All endpoints wrapped in try-except
- Errors logged and returned as HTTP 500

**Validation:**
- Pydantic models validate request schemas
- Automatic 422 responses for invalid requests

#### 3.9.4 CORS Configuration

**Middleware:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # All origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Note:** Should restrict `allow_origins` to specific domains

---

## 4. Implementation Details

### 4.1 Code Structure and Organization

**Directory Structure:**
```
mist/
├── config/              # YAML configuration files
├── data/                # Data storage
│   ├── databases/       # SQLite databases
│   ├── embeddings/      # Embedding checkpoints
│   ├── feedback/        # Feedback database
│   ├── knowledge_graph.graphml
│   └── vector_store/    # ChromaDB data
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── api/             # FastAPI server
│   ├── embeddings/      # Embedding models
│   ├── feedback/        # Feedback system
│   ├── knowledge/      # Knowledge graph
│   ├── learning/       # Training/self-improvement
│   ├── llm/            # LLM providers
│   ├── retrieval/      # RAG components
│   └── paths.py        # Path management
└── tests/              # Unit tests
```

**Module Organization:**
- Each major component in separate module
- `__init__.py` files for package structure
- Centralized path management via `paths.py`

### 4.2 Key Design Patterns

**1. Factory Pattern:**
- `LLMProviderFactory`: Creates LLM provider instances
- Tries providers in fallback order

**2. Strategy Pattern:**
- LLM providers implement common interface
- Can switch providers without changing calling code

**3. Singleton Pattern:**
- `get_paths()`: Global paths instance
- Can be overridden for testing

**4. Template Method Pattern:**
- Prompt templates define structure
- LLM providers implement generation differently

**5. Repository Pattern:**
- `VectorStore`: Abstracts vector database operations
- `FeedbackCollector`: Abstracts feedback storage

### 4.3 Configuration Management Approach

**YAML-Based Configuration:**
- Three main config files:
  - `config/llm_config.yaml`: LLM provider settings
  - `config/embedding_config.yaml`: Embedding model settings
  - `config/retrieval_config.yaml`: Retrieval parameters

**Configuration Loading:**
- Components load configs on initialization
- Default path: `get_paths().config.{config_name}`
- Can override with custom path

**Environment Variables:**
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Path overrides: `MIST_CONFIG_DIR`, `MIST_DATA_DIR`

### 4.4 Path Management System

**Location:** `src/paths.py`

**Classes:**
- `Paths`: Main paths manager
- `ConfigPaths`: Configuration file paths
- `DataPaths`: Data directory paths
- `FeedbackPaths`: Feedback database path
- `EmbeddingPaths`: Embedding checkpoint paths
- `DatabasePaths`: Database file paths

**Auto-Detection:**
- Finds mist root by locating `src` directory
- Uses `pathlib.Path` for cross-platform compatibility

**Environment Overrides:**
- `MIST_CONFIG_DIR`: Override config directory
- `MIST_DATA_DIR`: Override data directory

**Usage:**
```python
from paths import get_paths

paths = get_paths()
config_path = paths.config.llm_config
db_path = paths.data.databases.diagnostic_db
```

### 4.5 Dependency Management

**Requirements:** `requirements.txt`

**Core ML/AI:**
- `sentence-transformers>=2.2.0`
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `chromadb>=0.4.0`

**LLM Providers:**
- `openai>=1.0.0`
- `anthropic>=0.7.0`
- `langchain>=0.1.0`

**Retrieval & Ranking:**
- `rank-bm25>=0.2.2` (not currently used, but available)

**Knowledge Graph:**
- `networkx>=3.0`
- `rdflib>=6.0` (not currently used, but available)

**API & Infrastructure:**
- `fastapi>=0.100.0`
- `uvicorn>=0.23.0`
- `pydantic>=2.0.0`
- `sqlalchemy>=2.0.0` (not currently used directly)

**Data Processing:**
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `pyyaml>=6.0`

**Utilities:**
- `python-dotenv>=1.0.0`
- `tqdm>=4.65.0`

### 4.6 Error Handling Strategies

**Component Level:**
- Try-except blocks around critical operations
- Log errors and continue with fallbacks where possible

**API Level:**
- HTTP exceptions for client errors (404, 422)
- HTTP 500 for server errors with error details

**Initialization:**
- Provider factory tries multiple providers before failing
- Components check for required files/directories

**Graceful Degradation:**
- Knowledge graph optional (if not enabled, KG scoring skipped)
- Re-ranking optional (if not enabled, skipped)
- Missing OBD parameters handled with defaults

### 4.7 Logging and Monitoring Considerations

**Current State:**
- Minimal logging (print statements)
- Error messages in exceptions

**Recommended Enhancements:**
- Use Python `logging` module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log API requests/responses
- Log training progress
- Monitor vector store performance
- Track LLM API usage/costs

---

## 5. Data Models & Schemas

### 5.1 API Request/Response Models

See Section 3.9.2 for detailed schemas.

### 5.2 Internal Data Structures

**Document Format (Vector Store):**
```python
{
    'id': str,
    'text': str,
    'title': str,
    'procedure_id': str,
    'procedure_name': str,
    'fault_codes': List[str],
    'ecu_category': str,
    'metadata': Dict
}
```

**Search Result Format:**
```python
{
    'id': str,
    'score': float,  # Similarity score
    'text': str,
    'title': str,
    'procedure_id': str,
    'procedure_name': str,
    'fault_codes': List[str],
    'ecu_category': str,
    'metadata': Dict,
    'combined_score': float,  # After ranking
    'rerank_score': float  # If re-ranking used
}
```

**Session Format (Feedback):**
```python
{
    'session_id': str,
    'fault_codes': List[str],
    'obd_data': Dict,
    'clarification_questions': List[str],
    'user_responses': List[str],
    'recommended_guides': List[Dict],
    'selected_guide': str,
    'explicit_rating': int,
    'repair_outcome': str,
    'conversation_corrections': List[Dict],
    'timestamp': str
}
```

### 5.3 Database Schemas

**BMW Diagnostic Database Tables:**

**XEP_FAULTCODES:**
- `ID`: Primary key
- `CODE`: Fault code string
- `ECUVARIANTID`: Foreign key to ECUs
- `TITLE_ENGB`: English title

**XEP_FAULTLABELS:**
- `ID`: Foreign key to fault codes
- `TITLE_ENGB`: English description

**XEP_ECUVARIANTS:**
- `ID`: Primary key
- `NAME`: ECU name
- `TITLE_ENGB`: English title

**XEP_INFOOBJECTS:**
- `ID`: Primary key
- `NAME`: Procedure name
- `TITLE_ENGB`: English title

**XEP_INFOSEGMENTS:**
- `ID`: Foreign key to info objects
- `CONTENT_ENGB`: Content text

**XEP_DIAGNOSISOBJECTS:**
- `ID`: Primary key
- `CONTROLID`: Control ID
- `NAME`: Diagnostic name
- `TITLE_ENGB`: English title

**XEP_REFDIAGOBJECTS:**
- `ID`: Foreign key to fault codes
- `DIAGNOSISOBJECTCONTROLID`: Foreign key to diagnostic objects
- `PRIORITY`: Priority value

**RG_ECUFAULT_DOCIDS:**
- `ECUFAULT_ID`: Foreign key to fault codes
- `INFOOBJECTID`: Foreign key to repair procedures

**XEP_REFDIAGNOSISTREE:**
- `ID`: Parent control ID
- `DIAGNOSISOBJECTCONTROLID`: Child control ID

**Feedback Database Schema:** See Section 3.7.2

### 5.4 Vector Store Document Format

**Payload Structure (ChromaDB):**
```python
{
    'text': str,  # Procedure content
    'title': str,  # Procedure title
    'procedure_id': str,  # Unique ID
    'procedure_name': str,  # Procedure name
    'fault_codes': List[str],  # Associated fault codes
    'ecu_category': str,  # ECU category
    'metadata': Dict  # Additional metadata
}
```

**Vector:**
- 768-dimensional float array
- Normalized (L2 norm = 1.0)

### 5.5 Knowledge Graph Node/Edge Structures

**Node Attributes:**
- `type`: Node type (fault_code, ecu, repair_procedure, diagnostic_object)
- `id`: Database ID
- `label`: Display label
- Type-specific attributes (code, name, title, etc.)

**Edge Attributes:**
- `relationship`: Relationship type
- `weight`: Edge weight (float)
- Type-specific attributes (priority, etc.)

---

## 6. Configuration Specifications

### 6.1 LLM Configuration Structure

**File:** `config/llm_config.yaml`

**Structure:**
```yaml
providers:
  primary: openai  # Primary provider name
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
      You are a diagnostic assistant...
    user_template: |
      Fault Codes: {fault_codes}
      OBD Data: {obd_data}
      Current Context: {context}
      
      Generate clarifying questions...

  query_expansion:
    system: |
      You are a query expansion assistant...
    user_template: |
      Original Query: {original_query}
      User Responses: {user_responses}
      Expand the query...
```

### 6.2 Embedding Configuration Structure

**File:** `config/embedding_config.yaml`

**Structure:**
```yaml
models:
  fault_code:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    max_length: 512
    device: auto  # auto, cpu, cuda
  
  obd_data:
    type: structured_json
    normalization: true
    dimension: 384
  
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

fine_tuning:
  enabled: true
  checkpoint_interval: 1000
  min_feedback_samples: 100
  validation_split: 0.2
```

### 6.3 Retrieval Configuration Structure

**File:** `config/retrieval_config.yaml`

**Structure:**
```yaml
vector_store:
  provider: chromadb
  collection_name: repair_guides
  distance_metric: cosine
  vector_size: 768

retrieval:
  initial_k: 50
  rerank_k: 10
  min_similarity: 0.5

ranking:
  embedding_similarity: 0.4
  kg_path_score: 0.3
  feedback_score: 0.2
  recency: 0.1

reranking:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10

knowledge_graph:
  enabled: true
  max_path_length: 3
  min_path_score: 0.3

query_expansion:
  enabled: true
  max_expansions: 3
  expansion_weight: 0.3
```

### 6.4 Environment Variables Required

**API Keys:**
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI)
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)

**Path Overrides (Optional):**
- `MIST_CONFIG_DIR`: Override config directory path
- `MIST_DATA_DIR`: Override data directory path

### 6.5 Default Values and Tuning Parameters

**Retrieval:**
- `initial_k`: 50 (initial retrieval count)
- `rerank_k`: 10 (re-ranking count)
- `min_similarity`: 0.5 (minimum similarity threshold)

**Ranking Weights:**
- Embedding similarity: 0.4
- KG path score: 0.3
- Feedback score: 0.2
- Recency: 0.1
- **Note:** Weights should sum to 1.0

**Training:**
- Batch size: 32
- Learning rate: 1e-5
- Epochs: 10
- Minimum feedback samples: 10 (hardcoded), 100 (configurable for fine-tuning)

**Knowledge Graph:**
- Max path length: 3
- Min path score: 0.3

**Clarification:**
- Max clarifications per session: 3
- Ambiguity threshold: 0.6 (top score)
- Score variance threshold: 0.01 (for ambiguity detection)

---

## 7. Integration Points

### 7.1 How Components Connect

**ConversationalRAG Orchestration:**
```
ConversationalRAG
    ├─→ MultiModalEncoder.encode() → embeddings
    ├─→ VectorStore.search() → candidates
    ├─→ Ranker.rank() → ranked candidates
    ├─→ LLMProvider.generate() → clarification questions
    ├─→ QueryExpander.expand_query() → expanded query
    └─→ KnowledgeGraphQuery (optional) → KG scores
```

**Feedback Integration:**
```
API Endpoints
    └─→ FeedbackCollector
            ├─→ SQLite storage
            └─→ FeedbackAnalyzer
                    └─→ EmbeddingTrainer (for training)
```

**Knowledge Graph Integration:**
```
KnowledgeGraphBuilder
    └─→ SQLite Database → NetworkX Graph
            └─→ GraphML file
                    └─→ KnowledgeGraphQuery
                            └─→ Ranker (for KG scoring)
```

### 7.2 Interface Contracts

**LLM Provider Interface:**
```python
class LLMProvider(ABC):
    def generate(messages: List[Dict[str, str]], **kwargs) -> str
    def generate_stream(messages: List[Dict[str, str]], **kwargs) -> Iterator[str]
```

**Encoder Interface:**
```python
class Encoder:
    def encode(inputs) -> torch.Tensor
    def get_dimension() -> int
```

**Vector Store Interface:**
```python
class VectorStore:
    def add_documents(embeddings, documents, batch_size=100)
    def search(query_embedding, top_k=10, filter_dict=None) -> List[Dict]
    def update_document(doc_id, embedding=None, payload=None)
    def delete_document(doc_id)
```

### 7.3 Data Transformation Points

**Fault Codes + OBD Data → Embedding:**
- `MultiModalEncoder.encode(fault_codes, obd_data) → 768-dim tensor`

**Embedding → Vector Store:**
- Tensor → NumPy array → ChromaDB vector

**Search Results → Rankings:**
- Vector store results → Ranker → Combined scores

**Feedback → Training Data:**
- SQLite feedback → FeedbackDataset → PyTorch DataLoader

### 7.4 External Dependencies

**APIs:**
- OpenAI API (if using OpenAI)
- Anthropic API (if using Anthropic)
- Ollama API (if using open-source)

**Databases:**
- SQLite (BMW diagnostic databases, feedback storage)
- ChromaDB (vector database)

**Libraries:**
- PyTorch (neural networks, training)
- sentence-transformers (text embeddings)
- NetworkX (knowledge graph)
- FastAPI (API server)
- ChromaDB Client (vector database)

---

## 8. Database Integration Notes

### 8.1 Current Database Usage

**BMW Diagnostic Databases:**
- **Diagnostic.sqlite**: Main diagnostic database
  - Location: `data/databases/Diagnostic.sqlite`
  - Used by: `KnowledgeGraphBuilder`
  - Tables: See Section 5.3

- **bmw_diagnostic_ml_ready.sqlite**: ML-ready database
  - Location: `data/databases/bmw_diagnostic_ml_ready.sqlite`
  - Used by: `index_repair_guides.py` (potentially)
  - May contain cleaned/preprocessed data

**Feedback Database:**
- **feedback.db**: SQLite database for feedback storage
  - Location: `data/feedback/feedback.db`
  - Created automatically
  - Schema: See Section 3.7.2

### 8.2 Database Schema Understanding

**Key Relationships:**
- Fault codes → ECUs (via `ECUVARIANTID`)
- Fault codes → Diagnostic objects (via `XEP_REFDIAGOBJECTS`)
- Fault codes → Repair procedures (via `RG_ECUFAULT_DOCIDS`)
- Diagnostic objects → Diagnostic objects (via `XEP_REFDIAGNOSISTREE`)

**Data Extraction Queries:**

**Fault Codes:**
```sql
SELECT fc.ID, fc.CODE, fl.TITLE_ENGB
FROM XEP_FAULTCODES fc
LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.ID
WHERE fc.CODE IS NOT NULL
```

**Repair Procedures:**
```sql
SELECT io.ID, io.NAME, io.TITLE_ENGB, tc.data_clean
FROM XEP_INFOOBJECTS io
LEFT JOIN XEP_INFOSEGMENTS iseg ON io.ID = iseg.ID
LEFT JOIN cleaned_text_content_engb tc ON iseg.CONTENT_ENGB = tc.id
WHERE (io.NAME LIKE '%repair%' OR io.NAME LIKE '%fix%' OR io.NAME LIKE '%replace%')
```

**Fault-Repair Mappings:**
```sql
SELECT fc.ID as fault_id, rfd.INFOOBJECTID as procedure_id
FROM XEP_FAULTCODES fc
JOIN RG_ECUFAULT_DOCIDS rfd ON fc.ID = rfd.ECUFAULT_ID
```

### 8.3 Enhanced Database Mapping Requirements

**⚠️ IMPORTANT NOTE FOR NEW IMPLEMENTATION:**

The new implementation should incorporate **additional database mappings from another repository**. This enhancement should:

1. **Extend Knowledge Graph:**
   - Add new node types if needed
   - Add new relationship types
   - Incorporate mappings from additional databases

2. **Enhance Vector Store Indexing:**
   - Index additional document types
   - Include mappings from new databases
   - Enrich metadata with new database fields

3. **Update Query Patterns:**
   - Support queries across multiple database sources
   - Combine mappings from BMW database and new database
   - Handle conflicts/priorities between sources

4. **Database Schema Integration:**
   - Understand schema of new database
   - Map fields between databases
   - Create unified data model

5. **Configuration:**
   - Add configuration for new database paths
   - Configure mapping priorities
   - Enable/disable database sources

**Placeholder for Enhanced Mapping:**
- Location: Extend `KnowledgeGraphBuilder` and `index_repair_guides.py`
- Integration point: After BMW database extraction, add new database extraction
- Merge strategy: Combine graphs/indexes, handle conflicts

---

## 9. Scripts & Utilities

### 9.1 Knowledge Graph Building Script

**Location:** `scripts/build_knowledge_graph.py`

**Purpose:** Extract relationships from BMW diagnostic database and build NetworkX knowledge graph

**Process:**
1. Connect to `Diagnostic.sqlite`
2. Extract nodes: fault codes, ECUs, repair procedures, diagnostic objects
3. Extract edges: relationships between nodes
4. Build NetworkX MultiDiGraph
5. Save as GraphML file

**Usage:**
```bash
python scripts/build_knowledge_graph.py
```

**Output:**
- `data/knowledge_graph.graphml`
- Statistics printed to console

**Dependencies:**
- `KnowledgeGraphBuilder` from `src/knowledge/graph_builder.py`
- BMW diagnostic database must exist

### 9.2 Indexing Scripts

**Location:** `scripts/index_repair_guides.py`

**Purpose:** Index repair guides from BMW database into vector store

**Process:**
1. Connect to `Diagnostic.sqlite`
2. Query repair procedures with content
3. Extract fault codes associated with each procedure
4. Encode procedures using `MultiModalEncoder`
5. Add to ChromaDB vector store

**Usage:**
```bash
python scripts/index_repair_guides.py
```

**Output:**
- Documents indexed in ChromaDB collection `repair_guides`
- Statistics printed to console

**Batch Processing:**
- Processes in batches of 50
- Shows progress bar

**Dependencies:**
- `MultiModalEncoder` from `src/embeddings/multimodal_encoder.py`
- `VectorStore` from `src/retrieval/vector_store.py`
- BMW diagnostic database must exist

### 9.3 Training Scripts

**Location:** `scripts/train_embeddings.py`

**Purpose:** Train/update embeddings from feedback data

**Process:**
1. Initialize `EmbeddingTrainer`
2. Check feedback statistics
3. Verify minimum samples (10)
4. Load feedback data
5. Train embeddings with contrastive learning
6. Save checkpoints

**Usage:**
```bash
python scripts/train_embeddings.py
```

**Output:**
- Checkpoints saved to `data/embeddings/checkpoints/checkpoint_epoch_{N}.pt`
- Training progress printed to console

**Requirements:**
- At least 10 feedback sessions
- Feedback with ratings or outcomes

**Dependencies:**
- `EmbeddingTrainer` from `src/learning/embedding_trainer.py`
- `FeedbackCollector` and `FeedbackAnalyzer`

### 9.4 Usage Patterns

**Initial Setup:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional

# 3. Build knowledge graph
python scripts/build_knowledge_graph.py

# 4. Index repair guides
python scripts/index_repair_guides.py

# 5. Start API server
python -m src.api.server
```

**After Collecting Feedback:**
```bash
# Train embeddings
python scripts/train_embeddings.py
```

**Production Deployment:**
- Use `uvicorn` with production settings
- Configure CORS properly
- Set up logging
- Use ChromaDB server instead of local file storage

---

## 10. Deployment & Operations

### 10.1 Setup Requirements

**System Requirements:**
- Python 3.8+
- CUDA-capable GPU (optional, for faster embeddings)
- Sufficient disk space for databases and vector store

**Python Environment:**
- Virtual environment recommended
- Install dependencies: `pip install -r requirements.txt`

**External Services:**
- ChromaDB (local file-based or server)
- LLM API access (OpenAI/Anthropic) or local Ollama

### 10.2 Initialization Sequence

**1. Configuration:**
- Review and update YAML config files
- Set environment variables for API keys

**2. Database Preparation:**
- Ensure BMW diagnostic databases are in `data/databases/`
- Run `build_knowledge_graph.py` to create knowledge graph

**3. Vector Store Initialization:**
- Run `index_repair_guides.py` to index repair guides
- Verify collection exists and has documents

**4. API Server:**
- Start server: `python -m src.api.server`
- Or: `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`

**5. Verification:**
- Check `/health` endpoint
- Test `/query` endpoint with sample data

### 10.3 Runtime Dependencies

**Required:**
- BMW diagnostic databases accessible
- Knowledge graph file exists (if KG enabled)
- Vector store initialized
- LLM provider accessible (API or local)

**Optional:**
- Feedback database (created automatically)
- Embedding checkpoints (for fine-tuned models)

### 10.4 Production Considerations

**API Server:**
- Use production ASGI server (Gunicorn + Uvicorn workers)
- Configure CORS properly (restrict origins)
- Set up logging
- Use environment variables for secrets
- Implement rate limiting
- Add authentication/authorization

**Vector Store:**
- Use ChromaDB server instead of local file storage
- Configure replication and backups
- Monitor disk usage

**Database:**
- Backup SQLite databases regularly
- Consider migrating to PostgreSQL for production
- Monitor database size and performance

**LLM Providers:**
- Monitor API usage and costs
- Implement caching for common queries
- Set up fallback chains properly
- Handle rate limits

**Monitoring:**
- Log API requests/responses
- Track embedding generation time
- Monitor vector search performance
- Track feedback collection rates
- Monitor training progress

**Scaling:**
- Vector store can be scaled horizontally (ChromaDB cluster)
- API server can be scaled with load balancer
- Consider caching frequently accessed embeddings
- Batch processing for indexing/training

---

## Appendix A: Key Constants and Defaults

**Embedding Dimensions:**
- Fault code encoder: 384
- OBD data encoder: 384
- Multi-modal output: 768

**Retrieval Defaults:**
- Initial K: 50
- Re-rank K: 10
- Final recommendations: 10
- Min similarity: 0.5

**Ranking Weights:**
- Embedding: 0.4
- KG: 0.3
- Feedback: 0.2
- Recency: 0.1

**Training Defaults:**
- Batch size: 32
- Learning rate: 1e-5
- Epochs: 10
- Min feedback samples: 10

**Clarification:**
- Max questions: 3
- Ambiguity threshold: 0.6
- Score variance threshold: 0.01

---

## Appendix B: File Path Reference

**Configuration Files:**
- `config/llm_config.yaml`
- `config/embedding_config.yaml`
- `config/retrieval_config.yaml`

**Data Files:**
- `data/databases/Diagnostic.sqlite`
- `data/databases/bmw_diagnostic_ml_ready.sqlite`
- `data/knowledge_graph.graphml`
- `data/feedback/feedback.db`
- `data/vector_store/` (ChromaDB data)
- `data/embeddings/checkpoints/` (training checkpoints)

**Source Code:**
- `src/api/` - API server
- `src/embeddings/` - Embedding models
- `src/feedback/` - Feedback system
- `src/knowledge/` - Knowledge graph
- `src/learning/` - Training/self-improvement
- `src/llm/` - LLM providers
- `src/retrieval/` - RAG components
- `src/paths.py` - Path management

**Scripts:**
- `scripts/build_knowledge_graph.py`
- `scripts/index_repair_guides.py`
- `scripts/train_embeddings.py`

---

## Document Revision History

- **Version 1.0**: Initial comprehensive knowledge transfer document
- **Date**: 2024
- **Purpose**: AI-driven code generation for enhanced implementation

---

## Notes for AI Code Generation

1. **Preserve Architecture**: Maintain the modular structure and component separation
2. **Configuration-Driven**: Keep YAML-based configuration approach
3. **Path Management**: Use centralized path management system
4. **Error Handling**: Implement robust error handling with graceful degradation
5. **Database Integration**: Extend database mappings as specified in Section 8.3
6. **Type Hints**: Use Python type hints for all functions
7. **Documentation**: Include docstrings for all classes and methods
8. **Testing**: Create unit tests for critical components
9. **Logging**: Implement proper logging throughout
10. **Performance**: Consider optimization opportunities (caching, batching, etc.)

---

**End of Document**

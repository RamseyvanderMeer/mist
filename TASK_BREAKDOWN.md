# Task Breakdown: MIST Implementation Plan Decomposition

This document breaks down the MIST Enhanced Mapping Layer Implementation Plan into isolated, implementable tasks for AI coding agents. Tasks are organized by dependency order, not phase order, to enable parallel development where possible.

## Dependency Graph Overview

```
[CONFIG] → [BASE_CLASSES] → [ENCODERS] → [VECTOR_STORE] → [RETRIEVAL] → [CONVERSATIONAL_RAG] → [API]
     ↓            ↓             ↓              ↓                ↓                ↓                ↓
[DB_SCHEMA]  [LLM_PROVIDERS] [KG_BUILDER] [RERANKER]    [FEEDBACK]      [SESSION_MGMT]   [INTEGRATION]
     ↓            ↓             ↓              ↓                ↓                ↓                ↓
[DB_LAYER]   [PROMPTS]    [KG_QUERY]    [RANKER]      [ANALYZER]     [QUERY_EXPAND]   [SCRIPTS]
                                                          ↓                ↓
                                                    [REWARD_MODEL]   [AMBIGUITY]
                                                          ↓                ↓
                                                    [TRAINER]        [CLARIFICATION]
                                                          ↓
                                                    [ACTIVE_LEARNING]
```

## Task Organization

Tasks are numbered with format: `[PHASE]-[COMPONENT]-[NUMBER]`

- **P0**: Foundation/Infrastructure (no dependencies)
- **P1**: Core Components (depend on P0)
- **P2**: Retrieval Pipeline (depend on P1)
- **P3**: Conversational RAG (depend on P2)
- **P4**: Self-Improvement (depend on P2, P3)
- **P5**: API & Integration (depend on P3, P4)
- **P6**: Testing & Refinement (depend on P5)

---

## P0: Foundation & Infrastructure Tasks

### P0-CFG-001: Create Configuration Files
**Dependencies**: None  
**Complexity**: Low

**Description**: Create all YAML configuration files with default values as specified in the implementation plan.

**Inputs**: 
- Implementation plan section 9 (Configuration Files)

**Outputs**:
- `config/embedding_config.yaml` (may already exist, verify completeness)
- `config/retrieval_config.yaml` (may already exist, verify completeness)
- `config/llm_config.yaml` (may already exist, verify completeness)
- `config/training_config.yaml` (may already exist, verify completeness)

**Acceptance Criteria**:
- [ ] All 4 config files exist with complete structure
- [ ] All required fields from implementation plan are present
- [ ] Default values match specification
- [ ] YAML syntax is valid (can be loaded by PyYAML)
- [ ] Config files are documented with comments

**Files to Create/Modify**:
- `config/embedding_config.yaml`
- `config/retrieval_config.yaml`
- `config/llm_config.yaml`
- `config/training_config.yaml`

**Test Strategy**: 
- Unit test: Load each config file and verify structure
- Verify all required keys exist
- Test with config loader utility

---

### P0-DB-002: Create Database Schema Extensions
**Dependencies**: None  
**Complexity**: Medium

**Description**: Create SQL migration scripts to add new tables for MIST system (feedback_sessions, mist_embeddings, mist_feedback, mist_training_checkpoints).

**Inputs**:
- Implementation plan section 8 (Database Schema Extensions)
- Existing BMW ISTA database structure

**Outputs**:
- `scripts/migrations/create_mist_tables.sql`
- `src/database/schema.py` (SQLAlchemy models)
- `src/database/migrations.py` (migration utilities)

**Acceptance Criteria**:
- [ ] All 4 tables created with correct schema
- [ ] Foreign key relationships properly defined
- [ ] Indexes created for performance
- [ ] Migration script can be run idempotently
- [ ] SQLAlchemy models match SQL schema
- [ ] Schema can be validated against existing database

**Files to Create/Modify**:
- `scripts/migrations/create_mist_tables.sql`
- `src/database/schema.py` (new)
- `src/database/migrations.py` (new)
- `src/database/__init__.py` (new)

**Test Strategy**:
- Unit test: Create in-memory SQLite database and run migrations
- Verify all tables exist with correct columns
- Test foreign key constraints
- Test index creation

---

### P0-PATHS-003: Verify Path Management System
**Dependencies**: None  
**Complexity**: Low

**Description**: Verify that `src/paths.py` exists and handles all required paths. Extend if needed.

**Inputs**:
- Existing `src/paths.py`
- Implementation plan directory structure

**Outputs**:
- Verified/updated `src/paths.py`

**Acceptance Criteria**:
- [ ] All paths from implementation plan are accessible
- [ ] Environment variable overrides work
- [ ] Paths resolve correctly on different OS
- [ ] Missing directories are created automatically (optional)

**Files to Create/Modify**:
- `src/paths.py` (verify/extend)

**Test Strategy**:
- Unit test: Test all path properties
- Test environment variable overrides
- Test path resolution on different platforms

---

### P0-REQ-004: Verify and Update Requirements
**Dependencies**: None  
**Complexity**: Low

**Description**: Verify `requirements.txt` includes all dependencies from implementation plan. Add missing ones.

**Inputs**:
- Existing `requirements.txt`
- Implementation plan dependencies

**Outputs**:
- Updated `requirements.txt`

**Acceptance Criteria**:
- [ ] All required packages listed with version constraints
- [ ] No conflicting version requirements
- [ ] Optional dependencies clearly marked
- [ ] Requirements install successfully in clean environment

**Files to Create/Modify**:
- `requirements.txt`

**Test Strategy**:
- Create fresh virtual environment
- Install requirements
- Verify no import errors for core packages

---

## P1: Core Component Tasks

### P1-LLM-001: Create LLM Provider Abstract Base Class
**Dependencies**: P0-CFG-001  
**Complexity**: Medium

**Description**: Create abstract base class `LLMProvider` that defines interface for LLM operations (generate, stream, etc.).

**Inputs**:
- `config/llm_config.yaml`
- Implementation plan section 2.2.3

**Outputs**:
- `src/llm/provider.py` with `LLMProvider` ABC
- `src/llm/__init__.py` exports

**Acceptance Criteria**:
- [ ] Abstract base class defines required methods: `generate()`, `stream()`, `get_model_info()`
- [ ] Type hints for all methods
- [ ] Docstrings explain interface contract
- [ ] Raises `NotImplementedError` for abstract methods
- [ ] Includes error handling interface
- [ ] Supports async operations (optional but recommended)

**Files to Create/Modify**:
- `src/llm/provider.py`

**Test Strategy**:
- Unit test: Cannot instantiate abstract class
- Test that subclasses must implement all methods
- Mock test: Verify interface contract

---

### P1-LLM-002: Implement OpenAI Client
**Dependencies**: P1-LLM-001, P0-CFG-001  
**Complexity**: Medium

**Description**: Implement `OpenAIClient` class that extends `LLMProvider` for OpenAI API integration.

**Inputs**:
- `src/llm/provider.py`
- `config/llm_config.yaml`
- OpenAI API documentation

**Outputs**:
- `src/llm/openai_client.py` with `OpenAIClient` class

**Acceptance Criteria**:
- [ ] Implements all abstract methods from `LLMProvider`
- [ ] Loads config from YAML
- [ ] Handles API key from environment variable
- [ ] Implements retry logic with exponential backoff
- [ ] Handles rate limiting
- [ ] Proper error handling and logging
- [ ] Supports streaming responses
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/llm/openai_client.py`

**Test Strategy**:
- Unit test: Mock OpenAI API calls
- Test error handling (rate limits, API errors)
- Test config loading
- Integration test: Real API call (optional, requires API key)

---

### P1-LLM-003: Implement Anthropic Client
**Dependencies**: P1-LLM-001, P0-CFG-001  
**Complexity**: Medium

**Description**: Implement `AnthropicClient` class that extends `LLMProvider` for Anthropic API integration.

**Inputs**:
- `src/llm/provider.py`
- `config/llm_config.yaml`
- Anthropic API documentation

**Outputs**:
- `src/llm/anthropic_client.py` with `AnthropicClient` class

**Acceptance Criteria**:
- [ ] Implements all abstract methods from `LLMProvider`
- [ ] Loads config from YAML
- [ ] Handles API key from environment variable
- [ ] Implements retry logic
- [ ] Handles rate limiting
- [ ] Proper error handling and logging
- [ ] Supports streaming responses
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/llm/anthropic_client.py`

**Test Strategy**:
- Unit test: Mock Anthropic API calls
- Test error handling
- Test config loading
- Integration test: Real API call (optional)

---

**NOTE - Implementation Deviation**: Instead of implementing OpenAI (P1-LLM-002) and Anthropic (P1-LLM-003) clients as originally planned, a **Gemini LLM client** was implemented following the same pattern. The Gemini client (`src/llm/gemini_client.py`) implements all required `LLMProvider` abstract methods and integrates with the provider factory. OpenAI and Anthropic clients already exist in the codebase (`src/llm/openai_client.py` and `src/llm/anthropic_client.py`), so they were not re-implemented. The Gemini provider has been configured as the primary provider in `config/llm_config.yaml` with OpenAI and Anthropic as fallbacks.

---

### P1-LLM-004: Implement Prompt Template Manager
**Dependencies**: P0-CFG-001  
**Complexity**: Low

**Description**: Create `PromptTemplates` class to load and manage prompt templates from config files.

**Inputs**:
- `config/llm_config.yaml` (prompts section)
- Implementation plan section 9.3

**Outputs**:
- `src/llm/prompt_templates.py` with `PromptTemplates` class

**Acceptance Criteria**:
- [ ] Loads templates from YAML config
- [ ] Supports template variable substitution (using `{variable}` syntax)
- [ ] Methods for each template type: `get_clarification_prompt()`, `get_query_expansion_prompt()`
- [ ] Validates required variables are provided
- [ ] Handles missing templates gracefully
- [ ] Type hints and docstrings

**Files to Create/Modify**:
- `src/llm/prompt_templates.py`

**Test Strategy**:
- Unit test: Load templates from config
- Test variable substitution
- Test missing variable handling
- Test missing template handling

---

### P1-EMB-005: Implement Fault Code Encoder (Verify/Complete)
**Dependencies**: P0-CFG-001, P0-PATHS-003  
**Complexity**: High

**Description**: Verify `FaultCodeEncoder` implementation matches specification. Complete if missing features.

**Inputs**:
- Existing `src/embeddings/fault_code_encoder.py`
- `config/embedding_config.yaml`
- Implementation plan section 4.2.1

**Outputs**:
- Complete `src/embeddings/fault_code_encoder.py`

**Acceptance Criteria**:
- [ ] Uses E5-Mistral-7B-Instruct model
- [ ] Projects from 4096-dim to 768-dim
- [ ] Adds "query:" prefix for E5-Mistral
- [ ] Supports batch encoding
- [ ] L2 normalization option
- [ ] Fallback to smaller model if E5-Mistral fails
- [ ] Proper error handling and logging
- [ ] Type hints throughout
- [ ] Unit tests pass

**Files to Create/Modify**:
- `src/embeddings/fault_code_encoder.py`

**Test Strategy**:
- Unit test: Encode single and batch texts
- Test normalization
- Test fallback mechanism
- Test dimension output (768)
- Test instruction prefix addition

---

### P1-EMB-006: Implement OBD Data Encoder
**Dependencies**: P0-CFG-001  
**Complexity**: High

**Description**: Implement `OBDDataEncoder` neural network for structured OBD data encoding.

**Inputs**:
- `config/embedding_config.yaml`
- Implementation plan section 4.2.2

**Outputs**:
- `src/embeddings/obd_data_encoder.py` with `OBDDataEncoder` class

**Acceptance Criteria**:
- [ ] PyTorch `nn.Module` subclass
- [ ] Feature extraction layers (input_dim → hidden_dim → hidden_dim*2)
- [ ] Multi-head attention mechanism (8 heads)
- [ ] Output projection to 768-dim
- [ ] `normalize_obd_data()` method for parameter normalization
- [ ] Handles single dict or list of dicts
- [ ] Supports temporal patterns (if multiple readings)
- [ ] L2 normalization on output
- [ ] Proper initialization (Xavier/Kaiming)
- [ ] Type hints and docstrings

**Files to Create/Modify**:
- `src/embeddings/obd_data_encoder.py`

**Test Strategy**:
- Unit test: Forward pass with sample OBD data
- Test normalization function
- Test batch processing
- Test output dimension (768)
- Test attention mechanism
- Test with missing parameters

---

### P1-EMB-007: Implement Multi-Modal Encoder (Cross-Attention Fusion)
**Dependencies**: P1-EMB-005, P1-EMB-006  
**Complexity**: High

**Description**: Implement `MultiModalEncoder` that fuses fault code and OBD embeddings using cross-attention.

**Inputs**:
- `src/embeddings/fault_code_encoder.py`
- `src/embeddings/obd_data_encoder.py`
- `config/embedding_config.yaml`
- Implementation plan section 2.2.1

**Outputs**:
- `src/embeddings/multimodal_encoder.py` with `MultiModalEncoder` class

**Acceptance Criteria**:
- [ ] Combines `FaultCodeEncoder` and `OBDDataEncoder`
- [ ] Bidirectional cross-attention (8 heads)
- [ ] Residual connections
- [ ] Outputs unified 768-dim embedding
- [ ] Handles missing OBD data gracefully
- [ ] Supports encoding fault codes only (fallback)
- [ ] Proper initialization
- [ ] Type hints and docstrings

**Files to Create/Modify**:
- `src/embeddings/multimodal_encoder.py`

**Test Strategy**:
- Unit test: Encode with both inputs
- Test with missing OBD data
- Test output dimension (768)
- Test cross-attention mechanism
- Test residual connections

---

### P1-DB-008: Create Database Connection Layer
**Dependencies**: P0-DB-002  
**Complexity**: Medium

**Description**: Create database connection utilities and wrapper for BMW ISTA databases.

**Inputs**:
- `scripts/migrations/create_mist_tables.sql`
- Existing BMW ISTA database structure
- Implementation plan section 6.1

**Outputs**:
- `src/database/connection.py` (database connection manager)
- `src/database/ista_db.py` (BMW ISTA database wrapper)

**Acceptance Criteria**:
- [ ] SQLAlchemy connection management
- [ ] Context manager for connections
- [ ] Query utilities for BMW ISTA tables
- [ ] Methods for fault codes, ECUs, repair procedures
- [ ] Error handling and logging
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/database/connection.py` (new)
- `src/database/ista_db.py` (new)

**Test Strategy**:
- Unit test: Connect to test database
- Test query methods
- Test error handling
- Integration test: Query real ISTA database (if available)

---

### P1-VEC-009: Implement Vector Store Interface
**Dependencies**: P0-CFG-001, P0-PATHS-003  
**Complexity**: Medium

**Description**: Create `VectorStore` class that wraps Qdrant client with MIST-specific interface.

**Inputs**:
- `config/retrieval_config.yaml`
- Qdrant client library
- Implementation plan section 2.2.2

**Outputs**:
- `src/retrieval/vector_store.py` with `VectorStore` class

**Acceptance Criteria**:
- [ ] Initializes Qdrant collection (creates if not exists)
- [ ] Methods: `add()`, `search()`, `delete()`, `update()`
- [ ] Supports metadata filtering
- [ ] Handles 768-dim vectors
- [ ] Cosine similarity distance
- [ ] Proper error handling
- [ ] Connection pooling/reuse
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/vector_store.py`

**Test Strategy**:
- Unit test: Mock Qdrant client
- Test collection creation
- Test add/search/delete operations
- Test metadata filtering
- Integration test: Real Qdrant instance (optional)

---

## P2: Retrieval Pipeline Tasks

### P2-KG-001: Implement Knowledge Graph Builder
**Dependencies**: P1-DB-008  
**Complexity**: High

**Description**: Create script to build NetworkX knowledge graph from BMW ISTA databases.

**Inputs**:
- `src/database/ista_db.py`
- BMW ISTA database files
- Implementation plan section 2.2.2

**Outputs**:
- `src/knowledge/graph_builder.py` with `KnowledgeGraphBuilder` class
- `scripts/build_knowledge_graph.py` (executable script)

**Acceptance Criteria**:
- [ ] Creates NetworkX MultiDiGraph
- [ ] Nodes: fault codes, ECUs, diagnostic objects, repair procedures
- [ ] Edges: affects_ecu, has_diagnostic, has_repair (with weights)
- [ ] Loads from BMW ISTA database tables
- [ ] Saves to GraphML format
- [ ] Handles missing relationships gracefully
- [ ] Progress logging
- [ ] Can rebuild incrementally

**Files to Create/Modify**:
- `src/knowledge/graph_builder.py`
- `scripts/build_knowledge_graph.py`

**Test Strategy**:
- Unit test: Build graph from sample data
- Test node/edge creation
- Test graph serialization
- Integration test: Build from real database

---

### P2-KG-002: Implement Knowledge Graph Query Interface
**Dependencies**: P2-KG-001  
**Complexity**: Medium

**Description**: Create `KnowledgeGraphQuery` class for querying the knowledge graph (path finding, scoring).

**Inputs**:
- `src/knowledge/graph_builder.py`
- Knowledge graph file
- Implementation plan section 2.2.2

**Outputs**:
- `src/knowledge/graph_query.py` with `KnowledgeGraphQuery` class

**Acceptance Criteria**:
- [ ] Loads graph from file
- [ ] `find_paths()` method (fault → repair procedure)
- [ ] `score_paths()` method (weighted path scoring)
- [ ] Supports max path length constraint
- [ ] Handles missing paths gracefully
- [ ] Efficient path algorithms (shortest path)
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/knowledge/graph_query.py`

**Test Strategy**:
- Unit test: Query test graph
- Test path finding
- Test path scoring
- Test edge cases (no paths, multiple paths)

---

### P2-RER-003: Implement Reranker Module
**Dependencies**: P0-CFG-001  
**Complexity**: Medium

**Description**: Create `Reranker` class supporting both Cohere API and local cross-encoder models.

**Inputs**:
- `config/retrieval_config.yaml`
- Cohere API (optional)
- Cross-encoder models (local)

**Outputs**:
- `src/retrieval/reranker.py` with `Reranker` class

**Acceptance Criteria**:
- [ ] Supports Cohere API reranking
- [ ] Supports local cross-encoder models (fallback)
- [ ] Configurable via YAML
- [ ] Methods: `rerank(query, documents, top_k)`
- [ ] Returns relevance scores (0-1)
- [ ] Handles API errors gracefully
- [ ] Batch processing support
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/reranker.py`

**Test Strategy**:
- Unit test: Mock Cohere API
- Test local model fallback
- Test scoring output range
- Test error handling
- Integration test: Real API call (optional)

---

### P2-FB-004: Implement Feedback Collector
**Dependencies**: P0-DB-002, P1-DB-008  
**Complexity**: Medium

**Description**: Create `FeedbackCollector` class for storing and retrieving feedback data.

**Inputs**:
- `src/database/schema.py`
- `src/database/connection.py`
- Implementation plan section 7.1

**Outputs**:
- `src/feedback/collector.py` with `FeedbackCollector` class

**Acceptance Criteria**:
- [ ] Stores feedback sessions (SQLite)
- [ ] Methods: `save_session()`, `get_session()`, `save_feedback()`
- [ ] Supports all feedback types (ratings, outcomes, corrections)
- [ ] `get_procedure_score()` method (aggregated feedback)
- [ ] Proper error handling
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/feedback/collector.py`

**Test Strategy**:
- Unit test: In-memory database
- Test all CRUD operations
- Test score aggregation
- Test data validation

---

### P2-RANK-005: Implement Combined Ranker
**Dependencies**: P2-KG-002, P2-RER-003, P2-FB-004  
**Complexity**: Medium

**Description**: Create `Ranker` class that combines embedding similarity, rerank scores, KG scores, and feedback scores.

**Inputs**:
- `src/knowledge/graph_query.py`
- `src/retrieval/reranker.py`
- `src/feedback/collector.py`
- `config/retrieval_config.yaml`
- Implementation plan section 4.2.3

**Outputs**:
- `src/retrieval/ranker.py` with `Ranker` class

**Acceptance Criteria**:
- [ ] Combines 4 score types with configurable weights
- [ ] Default weights: 0.4 embedding, 0.3 rerank, 0.2 KG, 0.1 feedback
- [ ] `rank()` method takes candidates and returns ranked list
- [ ] Handles missing scores gracefully (defaults)
- [ ] Normalizes scores if needed
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/ranker.py`

**Test Strategy**:
- Unit test: Combine scores with known values
- Test weight configuration
- Test missing score handling
- Test ranking order

---

### P2-RET-006: Implement Enhanced Retriever Orchestrator
**Dependencies**: P1-VEC-009, P2-RER-003, P2-KG-002, P2-RANK-005, P1-EMB-007  
**Complexity**: High

**Description**: Create `EnhancedRetriever` class that orchestrates the multi-stage retrieval pipeline.

**Inputs**:
- `src/retrieval/vector_store.py`
- `src/retrieval/reranker.py`
- `src/knowledge/graph_query.py`
- `src/retrieval/ranker.py`
- `src/embeddings/multimodal_encoder.py`
- `config/retrieval_config.yaml`
- Implementation plan section 4.2.3

**Outputs**:
- `src/retrieval/conversational_rag.py` (partial - retrieval part)
- Or separate: `src/retrieval/enhanced_retriever.py`

**Acceptance Criteria**:
- [ ] Stage 1: Vector search (top-K=100)
- [ ] Stage 2: Re-ranking (top-K=50)
- [ ] Stage 3: KG path scoring
- [ ] Stage 4: Combined scoring
- [ ] Returns top-K ranked results
- [ ] Handles errors at each stage
- [ ] Configurable via YAML
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/enhanced_retriever.py` (new, or extend conversational_rag.py)

**Test Strategy**:
- Unit test: Mock all dependencies
- Test each stage independently
- Test full pipeline
- Test error handling
- Integration test: End-to-end retrieval

---

## P3: Conversational RAG Tasks

### P3-AMB-001: Implement Ambiguity Detection Logic
**Dependencies**: P2-RET-006  
**Complexity**: Medium

**Description**: Create `AmbiguityDetector` class that identifies when clarification is needed.

**Inputs**:
- `config/retrieval_config.yaml` (clarification section)
- Implementation plan section 2.1 (ambiguity check)

**Outputs**:
- `src/retrieval/ambiguity_detector.py` (new)
- Or method in `conversational_rag.py`

**Acceptance Criteria**:
- [ ] Checks top score < threshold (0.65)
- [ ] Checks score variance < threshold (0.02)
- [ ] Checks missing critical OBD parameters
- [ ] Returns boolean + reason
- [ ] Configurable thresholds
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/ambiguity_detector.py` (new)

**Test Strategy**:
- Unit test: Test each ambiguity criterion
- Test threshold configurations
- Test edge cases

---

### P3-CLAR-002: Implement Clarification Question Generation
**Dependencies**: P1-LLM-002, P1-LLM-003, P1-LLM-004, P2-RET-006  
**Complexity**: Medium

**Description**: Create `ClarificationGenerator` class that uses LLM to generate clarifying questions.

**Inputs**:
- `src/llm/openai_client.py` or `anthropic_client.py`
- `src/llm/prompt_templates.py`
- `config/llm_config.yaml`
- Fault codes, OBD data, top candidates

**Outputs**:
- `src/retrieval/clarification_generator.py` (new)
- Or method in `conversational_rag.py`

**Acceptance Criteria**:
- [ ] Uses LLM provider (with fallback)
- [ ] Loads prompt template from config
- [ ] Formats input (fault codes, OBD data, candidates)
- [ ] Parses LLM output to extract questions (numbered/bulleted)
- [ ] Returns 1-3 questions
- [ ] Handles LLM errors gracefully
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/clarification_generator.py` (new)

**Test Strategy**:
- Unit test: Mock LLM provider
- Test prompt formatting
- Test question parsing
- Test error handling
- Integration test: Real LLM call (optional)

---

### P3-EXP-003: Implement Query Expansion
**Dependencies**: P1-LLM-002, P1-LLM-003, P1-LLM-004  
**Complexity**: Medium

**Description**: Create `QueryExpander` class that expands queries using LLM and user responses.

**Inputs**:
- `src/llm/openai_client.py` or `anthropic_client.py`
- `src/llm/prompt_templates.py`
- Original query, user responses

**Outputs**:
- `src/retrieval/query_expander.py` with `QueryExpander` class

**Acceptance Criteria**:
- [ ] Uses LLM provider
- [ ] Loads query expansion prompt template
- [ ] Formats original query + user responses
- [ ] Returns expanded query text
- [ ] Handles LLM errors gracefully
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/query_expander.py`

**Test Strategy**:
- Unit test: Mock LLM provider
- Test prompt formatting
- Test expansion output
- Test error handling

---

### P3-SESS-004: Implement Session Management
**Dependencies**: P0-DB-002, P1-DB-008  
**Complexity**: Medium

**Description**: Create `SessionManager` class for tracking multi-turn conversations.

**Inputs**:
- `src/database/schema.py`
- `src/database/connection.py`

**Outputs**:
- `src/retrieval/session_manager.py` (new)
- Or method in `conversational_rag.py`

**Acceptance Criteria**:
- [ ] Creates new sessions (UUID)
- [ ] Stores session state (fault codes, OBD data, questions, responses)
- [ ] Retrieves session by ID
- [ ] Updates session with responses
- [ ] Tracks clarification history
- [ ] Handles session expiration (optional)
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/session_manager.py` (new)

**Test Strategy**:
- Unit test: In-memory database
- Test session creation/retrieval
- Test state updates
- Test expiration (if implemented)

---

### P3-RAG-005: Implement Conversational RAG Orchestrator
**Dependencies**: P2-RET-006, P3-AMB-001, P3-CLAR-002, P3-EXP-003, P3-SESS-004  
**Complexity**: High

**Description**: Create main `ConversationalRAG` class that orchestrates the full conversational flow.

**Inputs**:
- All retrieval and conversational components
- `config/retrieval_config.yaml`
- `config/llm_config.yaml`

**Outputs**:
- `src/retrieval/conversational_rag.py` with `ConversationalRAG` class

**Acceptance Criteria**:
- [ ] `query()` method: fault codes + OBD data → recommendations or clarification
- [ ] `clarify()` method: session_id + responses → refined recommendations
- [ ] Integrates ambiguity detection
- [ ] Generates clarification questions when needed
- [ ] Expands query with user responses
- [ ] Manages sessions
- [ ] Returns structured response (dict with recommendations, questions, etc.)
- [ ] Proper error handling
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/retrieval/conversational_rag.py`

**Test Strategy**:
- Unit test: Mock all dependencies
- Test high-confidence path (no clarification)
- Test low-confidence path (clarification)
- Test multi-turn conversation
- Integration test: End-to-end flow

---

## P4: Self-Improvement Tasks

### P4-FB-001: Implement Feedback Analyzer
**Dependencies**: P2-FB-004  
**Complexity**: Medium

**Description**: Create `FeedbackAnalyzer` class for analyzing feedback statistics and trends.

**Inputs**:
- `src/feedback/collector.py`
- Feedback database

**Outputs**:
- `src/feedback/analyzer.py` with `FeedbackAnalyzer` class

**Acceptance Criteria**:
- [ ] Methods: `get_statistics()`, `get_procedure_ratings()`, `get_trends()`
- [ ] Calculates average ratings
- [ ] Identifies low-rated procedures
- [ ] Tracks feedback over time
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/feedback/analyzer.py`

**Test Strategy**:
- Unit test: Test with sample feedback data
- Test statistics calculation
- Test trend analysis

---

### P4-REWARD-002: Implement Reward Model
**Dependencies**: P2-FB-004  
**Complexity**: High

**Description**: Create `RewardModel` neural network for RLHF (predicts feedback scores from embeddings).

**Inputs**:
- `config/training_config.yaml`
- Implementation plan section 7.2

**Outputs**:
- `src/feedback/reward_model.py` with `RewardModel` class

**Acceptance Criteria**:
- [ ] PyTorch `nn.Module` subclass
- [ ] Architecture: 768 → 512 → 256 → 1
- [ ] Takes query + document embeddings
- [ ] Outputs reward signal (0-1)
- [ ] Proper initialization
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/feedback/reward_model.py`

**Test Strategy**:
- Unit test: Forward pass with sample embeddings
- Test output range (0-1)
- Test architecture layers

---

### P4-TRAIN-003: Implement Contrastive Loss Function
**Dependencies**: None  
**Complexity**: Medium

**Description**: Create contrastive learning loss function (InfoNCE) for embedding fine-tuning.

**Inputs**:
- Implementation plan section 7.3

**Outputs**:
- `src/learning/contrastive_trainer.py` (partial - loss function)
- Or separate: `src/learning/losses.py`

**Acceptance Criteria**:
- [ ] InfoNCE loss implementation
- [ ] Supports hard negative mining
- [ ] Configurable temperature parameter
- [ ] Handles variable number of negatives
- [ ] Proper gradient computation
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/learning/losses.py` (new)
- Or `src/learning/contrastive_trainer.py`

**Test Strategy**:
- Unit test: Test loss computation
- Test gradient flow
- Test with different numbers of negatives

---

### P4-TRAIN-004: Implement Embedding Trainer
**Dependencies**: P1-EMB-007, P2-FB-004, P4-TRAIN-003  
**Complexity**: High

**Description**: Create `EmbeddingTrainer` class for fine-tuning embeddings using contrastive learning.

**Inputs**:
- `src/embeddings/multimodal_encoder.py`
- `src/feedback/collector.py`
- `src/learning/losses.py` or `contrastive_trainer.py`
- `config/training_config.yaml`

**Outputs**:
- `src/embeddings/embedding_trainer.py` with `EmbeddingTrainer` class

**Acceptance Criteria**:
- [ ] Creates dataset from feedback (positive/negative pairs)
- [ ] Trains with contrastive loss
- [ ] Supports checkpointing
- [ ] Validation split
- [ ] Learning rate scheduling
- [ ] Progress logging
- [ ] Saves checkpoints after each epoch
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/embeddings/embedding_trainer.py`

**Test Strategy**:
- Unit test: Test dataset creation
- Test training loop (mock)
- Test checkpointing
- Integration test: Train on sample data

---

### P4-ACTIVE-005: Implement Active Learning Module
**Dependencies**: P2-RET-006, P2-FB-004  
**Complexity**: Medium

**Description**: Create `ActiveLearning` class for identifying uncertain cases for human review.

**Inputs**:
- `src/retrieval/enhanced_retriever.py`
- `config/retrieval_config.yaml`
- Implementation plan section 7.4

**Outputs**:
- `src/learning/active_learning.py` with `ActiveLearning` class

**Acceptance Criteria**:
- [ ] `identify_uncertain_cases()` method
- [ ] Uses entropy-based or score variance
- [ ] Returns list of uncertain cases
- [ ] Configurable thresholds
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/learning/active_learning.py`

**Test Strategy**:
- Unit test: Test uncertainty detection
- Test threshold configurations
- Test edge cases

---

## P5: API & Integration Tasks

### P5-API-001: Create API Schemas (Pydantic Models)
**Dependencies**: None  
**Complexity**: Low

**Description**: Define Pydantic models for API request/response schemas.

**Inputs**:
- API requirements from implementation plan

**Outputs**:
- `src/api/schemas.py` with request/response models

**Acceptance Criteria**:
- [ ] `QueryRequest` model (fault_codes, obd_data, vehicle_context)
- [ ] `QueryResponse` model (recommendations, clarification_questions, session_id)
- [ ] `ClarifyRequest` model (session_id, responses)
- [ ] `FeedbackRequest` model (session_id, rating, outcome, etc.)
- [ ] Proper validation
- [ ] Type hints throughout

**Files to Create/Modify**:
- `src/api/schemas.py`

**Test Strategy**:
- Unit test: Test model validation
- Test serialization/deserialization

---

### P5-API-002: Implement FastAPI Server
**Dependencies**: P3-RAG-005, P5-API-001, P2-FB-004  
**Complexity**: High

**Description**: Create FastAPI server with endpoints for query, clarify, and feedback.

**Inputs**:
- `src/retrieval/conversational_rag.py`
- `src/api/schemas.py`
- `src/feedback/collector.py`

**Outputs**:
- `src/api/server.py` with FastAPI app

**Acceptance Criteria**:
- [ ] `/query` endpoint (POST)
- [ ] `/clarify` endpoint (POST)
- [ ] `/feedback` endpoint (POST)
- [ ] `/feedback/{session_id}` endpoint (GET)
- [ ] Proper error handling (HTTP status codes)
- [ ] Request validation
- [ ] Response serialization
- [ ] CORS configuration (if needed)
- [ ] Health check endpoint

**Files to Create/Modify**:
- `src/api/server.py`

**Test Strategy**:
- Unit test: Test each endpoint with mock dependencies
- Test error handling
- Integration test: Start server and test endpoints

---

### P5-SCRIPT-003: Create Indexing Script for Repair Guides
**Dependencies**: P1-EMB-007, P1-VEC-009, P1-DB-008  
**Complexity**: Medium

**Description**: Create script to index repair guides from BMW ISTA database into vector store.

**Inputs**:
- `src/database/ista_db.py`
- `src/embeddings/multimodal_encoder.py`
- `src/retrieval/vector_store.py`
- BMW ISTA database

**Outputs**:
- `scripts/index_repair_guides.py`

**Acceptance Criteria**:
- [ ] Loads repair procedures from database
- [ ] Encodes procedure text (fault codes + procedure content)
- [ ] Stores embeddings in vector store
- [ ] Includes metadata (procedure_id, fault_codes, ECU, etc.)
- [ ] Progress logging
- [ ] Can resume interrupted indexing
- [ ] Handles errors gracefully

**Files to Create/Modify**:
- `scripts/index_repair_guides.py`

**Test Strategy**:
- Unit test: Test with sample procedures
- Test encoding and storage
- Test error handling
- Integration test: Index real database (if available)

---

### P5-SCRIPT-004: Create Training Script
**Dependencies**: P4-TRAIN-004  
**Complexity**: Low

**Description**: Create executable script for training embeddings.

**Inputs**:
- `src/embeddings/embedding_trainer.py`
- `config/training_config.yaml`

**Outputs**:
- `scripts/train_embeddings.py`

**Acceptance Criteria**:
- [ ] Loads config
- [ ] Initializes trainer
- [ ] Runs training loop
- [ ] Saves checkpoints
- [ ] Command-line arguments (config path, etc.)
- [ ] Progress logging

**Files to Create/Modify**:
- `scripts/train_embeddings.py`

**Test Strategy**:
- Unit test: Test script execution (mock training)
- Test config loading
- Test checkpoint saving

---

### P5-SCRIPT-005: Create Feedback Collection Utility
**Dependencies**: P2-FB-004  
**Complexity**: Low

**Description**: Create utility script for collecting/managing feedback data.

**Inputs**:
- `src/feedback/collector.py`

**Outputs**:
- `scripts/collect_feedback.py` (may already exist, verify)

**Acceptance Criteria**:
- [ ] Command-line interface for adding feedback
- [ ] Can export feedback data
- [ ] Can view feedback statistics
- [ ] Proper CLI argument parsing

**Files to Create/Modify**:
- `scripts/collect_feedback.py`

**Test Strategy**:
- Unit test: Test CLI commands
- Test data export/import

---

## P6: Testing & Refinement Tasks

### P6-TEST-001: Create Unit Tests for Embeddings
**Dependencies**: P1-EMB-005, P1-EMB-006, P1-EMB-007  
**Complexity**: Medium

**Description**: Write comprehensive unit tests for all embedding components.

**Inputs**:
- All embedding modules

**Outputs**:
- `tests/test_embeddings.py` (may already exist, extend)

**Acceptance Criteria**:
- [ ] Tests for `FaultCodeEncoder`
- [ ] Tests for `OBDDataEncoder`
- [ ] Tests for `MultiModalEncoder`
- [ ] Test edge cases (missing data, errors)
- [ ] Test output dimensions
- [ ] Test normalization
- [ ] High code coverage (>80%)

**Files to Create/Modify**:
- `tests/test_embeddings.py`

**Test Strategy**:
- Run pytest
- Verify coverage report

---

### P6-TEST-002: Create Unit Tests for Retrieval
**Dependencies**: P2-RET-006, P2-RER-003, P2-KG-002  
**Complexity**: Medium

**Description**: Write comprehensive unit tests for retrieval components.

**Inputs**:
- All retrieval modules

**Outputs**:
- `tests/test_retrieval.py` (may already exist, extend)

**Acceptance Criteria**:
- [ ] Tests for `VectorStore`
- [ ] Tests for `Reranker`
- [ ] Tests for `KnowledgeGraphQuery`
- [ ] Tests for `EnhancedRetriever`
- [ ] Test multi-stage pipeline
- [ ] Test error handling
- [ ] High code coverage

**Files to Create/Modify**:
- `tests/test_retrieval.py`

**Test Strategy**:
- Run pytest
- Verify coverage report

---

### P6-TEST-003: Create Integration Tests for Conversational RAG
**Dependencies**: P3-RAG-005  
**Complexity**: High

**Description**: Write end-to-end integration tests for conversational RAG flow.

**Inputs**:
- `src/retrieval/conversational_rag.py`
- Mock/test data

**Outputs**:
- `tests/test_conversational_rag.py` (may already exist, extend)

**Acceptance Criteria**:
- [ ] Test high-confidence query (no clarification)
- [ ] Test low-confidence query (with clarification)
- [ ] Test multi-turn conversation
- [ ] Test error scenarios
- [ ] Uses test fixtures/mocks

**Files to Create/Modify**:
- `tests/test_conversational_rag.py`

**Test Strategy**:
- Run pytest
- Verify all scenarios pass

---

### P6-PERF-004: Performance Profiling and Optimization
**Dependencies**: P5-API-002  
**Complexity**: High

**Description**: Profile system performance and optimize bottlenecks.

**Inputs**:
- Complete system
- Performance requirements

**Outputs**:
- Performance report
- Optimized code

**Acceptance Criteria**:
- [ ] Profile embedding generation time
- [ ] Profile retrieval time
- [ ] Profile API response time
- [ ] Identify bottlenecks
- [ ] Optimize critical paths
- [ ] Document performance metrics

**Files to Create/Modify**:
- Various (based on profiling results)

**Test Strategy**:
- Benchmark before/after optimization
- Verify performance improvements

---

## Parallelization Opportunities

### Can Be Developed in Parallel (No Dependencies):

**Group 1 - Foundation (P0)**:
- P0-CFG-001 (Config files)
- P0-DB-002 (Database schema)
- P0-PATHS-003 (Path management)
- P0-REQ-004 (Requirements)

**Group 2 - LLM Providers (P1)**:
- P1-LLM-002 (OpenAI client) - *Already exists in codebase*
- P1-LLM-003 (Anthropic client) - *Already exists in codebase*
- Gemini LLM client - *Implemented instead* (follows same pattern as P1-LLM-002/P1-LLM-003)
- P1-LLM-004 (Prompt templates)

**Group 3 - Encoders (P1)**:
- P1-EMB-005 (Fault code encoder)
- P1-EMB-006 (OBD encoder)

**Group 4 - Independent Components**:
- P1-VEC-009 (Vector store)
- P1-DB-008 (Database layer)
- P2-FB-004 (Feedback collector)

### Sequential Dependencies:

**Critical Path 1 - Retrieval Pipeline**:
P0-CFG-001 → P1-EMB-005 → P1-EMB-006 → P1-EMB-007 → P1-VEC-009 → P2-RET-006

**Critical Path 2 - Conversational RAG**:
P1-LLM-001 → P1-LLM-002 → P1-LLM-004 → P3-CLAR-002 → P3-RAG-005

**Critical Path 3 - Self-Improvement**:
P2-FB-004 → P4-TRAIN-003 → P4-TRAIN-004

## Critical Path Summary

The longest dependency chain that blocks the most work:

1. **Configuration** (P0-CFG-001) - Blocks everything
2. **Encoders** (P1-EMB-005 → P1-EMB-006 → P1-EMB-007) - Blocks retrieval
3. **Retrieval Pipeline** (P2-RET-006) - Blocks conversational RAG
4. **Conversational RAG** (P3-RAG-005) - Blocks API
5. **API** (P5-API-002) - Blocks integration testing

**Estimated Timeline** (assuming parallel work where possible):
- P0: 1-2 days
- P1: 5-7 days (parallel work)
- P2: 7-10 days
- P3: 5-7 days
- P4: 7-10 days (can overlap with P3)
- P5: 5-7 days
- P6: 5-7 days

**Total: ~6-8 weeks** (with parallelization)

---

## Task Implementation Notes

### For AI Coding Agents:

1. **Start with P0 tasks** - These have no dependencies and enable all other work
2. **Work in parallel groups** - Many tasks can be done simultaneously
3. **Create interfaces first** - Abstract base classes enable parallel development
4. **Test incrementally** - Write tests as you implement, don't wait
5. **Use mocks** - Mock dependencies to test components independently
6. **Follow type hints** - All code should have complete type annotations
7. **Document as you go** - Docstrings for all classes and methods
8. **Handle errors gracefully** - All components should have proper error handling

### Validation Checklist for Each Task:

- [ ] Code follows project structure
- [ ] Type hints throughout
- [ ] Docstrings for all public methods
- [ ] Error handling implemented
- [ ] Logging added where appropriate
- [ ] Unit tests written
- [ ] Tests pass
- [ ] No linter errors
- [ ] Configurable via YAML (where applicable)
- [ ] Dependencies properly declared

---

**Document Version**: 1.0  
**Created**: 2024  
**Status**: Ready for Implementation

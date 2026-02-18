# MIST Data Optimization Implementation Plan

## Overview

This plan outlines how to use the scraped web data to optimize the MIST system's accuracy. The plan includes data processing, integration, training pipeline modifications, and evaluation strategies.

## Current System Limitations

Based on codebase analysis, the system currently:
1. Relies primarily on BMW ISTA proprietary databases
2. Has limited training data from user feedback (requires minimum 10-100 samples)
3. Uses pre-trained embeddings (E5-Mistral-7B-Instruct) without domain-specific fine-tuning
4. Lacks diverse real-world fault code + OBD data + repair outcome mappings

## Optimization Strategy

### Phase 1: Data Processing & Integration (Week 1-2)

#### 1.1 Data Validation Pipeline

**File**: `scripts/process_scraped_data.py`

**Purpose**: Clean, validate, and normalize scraped web data

**Key Functions**:
```python
def validate_fault_code(code: str) -> bool:
    """Validate fault code format (P-codes, manufacturer codes)"""
    
def normalize_obd_data(obd_data: dict) -> dict:
    """Normalize OBD-II data to standard format"""
    
def extract_repair_steps(text: str) -> list:
    """Extract structured repair steps from free text"""
    
def calculate_quality_score(record: dict) -> float:
    """Calculate data quality score (0.0-1.0)"""
    
def deduplicate_records(records: list) -> list:
    """Remove duplicate records based on fault codes + vehicle context"""
```

**Output**: Cleaned JSONL files in `data/training/scraped/`

#### 1.2 Data Schema Mapping

**File**: `scripts/map_scraped_to_mist_schema.py`

**Purpose**: Map scraped data to MIST's internal schema

**Mapping**:
- Scraped `fault_codes` → MIST `fault_codes` (List[str])
- Scraped `obd_data` → MIST `obd_data` (Dict[str, float])
- Scraped `repair_guide` → MIST repair guide format
- Scraped `vehicle_context` → MIST `vehicle_context`
- Scraped `outcome` → MIST feedback format

**Integration Points**:
- Store in `mist_data.db` FeedbackSession table
- Create synthetic feedback records for training
- Link to existing repair guide embeddings

#### 1.3 Data Enrichment

**File**: `scripts/enrich_scraped_data.py`

**Purpose**: Enhance scraped data with additional context

**Enrichment Steps**:
1. **Fault Code Expansion**: Add fault code descriptions from BMW ISTA database
2. **OBD Data Completion**: Use statistical imputation for missing OBD parameters
3. **Repair Guide Linking**: Match scraped repair guides to existing ISTA procedures
4. **Vehicle Context Enhancement**: Add engine codes, model codes from vehicle database
5. **Knowledge Graph Linking**: Connect to existing knowledge graph nodes

### Phase 2: Training Data Preparation (Week 2-3)

#### 2.1 Contrastive Learning Dataset Creation

**File**: `scripts/create_training_dataset.py`

**Purpose**: Create training pairs from scraped data

**Process**:
1. **Positive Pairs**: (fault_codes + obd_data, correct_repair_guide)
2. **Negative Pairs**: (fault_codes + obd_data, incorrect_repair_guides)
3. **Hard Negative Mining**: Select negatives from top-K retrieved but not selected guides

**Output Format**:
```python
{
    "anchor": embedding,  # Query embedding (fault_codes + obd_data)
    "positive": embedding,  # Correct repair guide embedding
    "negatives": [embedding, ...],  # Incorrect guide embeddings
    "metadata": {
        "fault_codes": [...],
        "vehicle_context": {...},
        "outcome": "success"
    }
}
```

#### 2.2 Data Splitting Strategy

**Split Ratios**:
- Training: 70%
- Validation: 15%
- Test: 15%

**Stratification**:
- Stratify by fault code category (P0xxx, P1xxx, P2xxx, P3xxx)
- Stratify by vehicle make/model
- Stratify by outcome (success/failure)

**File**: `scripts/split_training_data.py`

#### 2.3 Synthetic Data Generation

**File**: `scripts/generate_synthetic_data.py`

**Purpose**: Generate additional training data through augmentation

**Augmentation Techniques**:
1. **OBD Data Variation**: Add noise to OBD readings (±5% variation)
2. **Fault Code Combinations**: Create combinations of related codes
3. **Vehicle Context Variation**: Vary year, mileage within reasonable ranges
4. **Text Paraphrasing**: Rephrase repair guide descriptions (using LLM)

### Phase 3: Model Training Enhancements (Week 3-4)

#### 3.1 Enhanced Embedding Trainer

**File**: `src/embeddings/enhanced_embedding_trainer.py`

**Modifications to Existing Trainer**:

1. **Multi-Source Data Loading**:
```python
def load_training_data(self, sources: List[str]):
    """
    Load data from multiple sources:
    - User feedback (existing)
    - Scraped web data (new)
    - Synthetic data (new)
    """
```

2. **Weighted Sampling**:
```python
def create_weighted_dataset(self, datasets: Dict[str, Dataset], weights: Dict[str, float]):
    """
    Combine datasets with weights:
    - User feedback: weight 1.0 (highest quality)
    - Scraped data: weight 0.7 (medium quality)
    - Synthetic data: weight 0.3 (lowest quality)
    """
```

3. **Curriculum Learning**:
```python
def create_curriculum(self, dataset: Dataset):
    """
    Order training by difficulty:
    - Easy: Single fault code, clear symptoms
    - Medium: Multiple codes, some ambiguity
    - Hard: Complex cases, ambiguous symptoms
    """
```

#### 3.2 Domain-Specific Fine-Tuning

**File**: `scripts/fine_tune_fault_code_encoder.py`

**Purpose**: Fine-tune E5-Mistral encoder on automotive diagnostic text

**Process**:
1. Create text pairs from scraped data:
   - (fault_code_description, repair_procedure)
   - (symptoms, fault_code)
   - (obd_readings_description, diagnosis)

2. Fine-tune using contrastive learning:
```python
# Pseudo-code
for (text1, text2) in positive_pairs:
    emb1 = encoder(text1)
    emb2 = encoder(text2)
    loss = contrastive_loss(emb1, emb2, positive=True)

for (text1, text2) in negative_pairs:
    emb1 = encoder(text1)
    emb2 = encoder(text2)
    loss = contrastive_loss(emb1, emb2, positive=False)
```

3. Save fine-tuned model: `models/fine_tuned_e5_mistral_automotive/`

#### 3.3 OBD Data Encoder Enhancement

**File**: `src/embeddings/enhanced_obd_encoder.py`

**Enhancements**:
1. **Feature Engineering**: Add derived features from OBD data
   - Fuel trim ratios
   - Temperature differentials
   - Pressure ratios
   - Timing variations

2. **Attention Mechanisms**: Add attention to important OBD parameters
   - Learn which parameters are most relevant for each fault code
   - Weight parameters based on fault code type

3. **Temporal Patterns**: If temporal data available, add LSTM/GRU layers

### Phase 4: Retrieval System Optimization (Week 4-5)

#### 4.1 Enhanced Retriever with Learned Weights

**File**: `src/retrieval/optimized_retriever.py`

**Enhancements**:
1. **Learned Scoring Weights**: Train a small MLP to combine:
   - Embedding similarity
   - Knowledge graph scores
   - Feedback scores
   - Recency scores

2. **Fault Code Specific Retrieval**:
```python
def retrieve_by_fault_category(self, fault_codes: List[str], obd_data: dict):
    """
    Use fault code category to adjust retrieval:
    - P0xxx (fuel/air): Weight MAF, O2 sensors higher
    - P1xxx (transmission): Weight transmission parameters
    - P2xxx (emissions): Weight O2 sensors, EGR
    - P3xxx (ignition): Weight RPM, timing
    """
```

#### 4.2 Improved Reranking

**File**: `src/retrieval/enhanced_reranker.py`

**Enhancements**:
1. **Cross-Encoder Fine-Tuning**: Fine-tune cross-encoder on scraped data
2. **Fault Code Context**: Include fault code category in reranking
3. **Vehicle Context Awareness**: Adjust scores based on vehicle make/model

### Phase 5: Evaluation & Validation (Week 5-6)

#### 5.1 Evaluation Metrics

**File**: `scripts/evaluate_model.py`

**Metrics**:
1. **Retrieval Accuracy**:
   - Precision@K (K=1, 5, 10)
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG@10)

2. **Fault Code Specific Metrics**:
   - Accuracy by fault code category
   - Accuracy by vehicle make/model
   - Accuracy by OBD data availability

3. **Before/After Comparison**:
   - Compare metrics before and after training on scraped data
   - A/B testing on held-out test set

#### 5.2 Error Analysis

**File**: `scripts/analyze_errors.py`

**Purpose**: Identify failure modes and improvement opportunities

**Analysis**:
1. **Failure Cases**: Extract cases where correct guide not in top-K
2. **Pattern Analysis**: Identify common error patterns
3. **Data Gaps**: Identify fault codes/vehicles with insufficient data

### Phase 6: Continuous Improvement Pipeline (Week 6+)

#### 6.1 Active Learning Integration

**File**: `scripts/active_learning_pipeline.py`

**Process**:
1. Identify uncertain cases from scraped data
2. Prioritize cases with high information gain
3. Collect additional data for these cases
4. Retrain model with new data

#### 6.2 Feedback Loop

**File**: `scripts/feedback_integration.py`

**Process**:
1. Monitor model performance on real queries
2. Compare predictions to scraped data outcomes
3. Identify discrepancies
4. Update training data with corrections

## Implementation Files

### New Scripts to Create

1. `scripts/process_scraped_data.py` - Data validation and cleaning
2. `scripts/map_scraped_to_mist_schema.py` - Schema mapping
3. `scripts/enrich_scraped_data.py` - Data enrichment
4. `scripts/create_training_dataset.py` - Training dataset creation
5. `scripts/split_training_data.py` - Data splitting
6. `scripts/generate_synthetic_data.py` - Synthetic data generation
7. `scripts/fine_tune_fault_code_encoder.py` - Encoder fine-tuning
8. `scripts/evaluate_model.py` - Model evaluation
9. `scripts/analyze_errors.py` - Error analysis
10. `scripts/active_learning_pipeline.py` - Active learning
11. `scripts/feedback_integration.py` - Feedback integration

### Modified Files

1. `src/embeddings/embedding_trainer.py` - Add multi-source data loading
2. `src/retrieval/enhanced_retriever.py` - Add learned weights
3. `src/retrieval/reranker.py` - Enhance reranking
4. `config/training_config.yaml` - Add new training parameters

### New Configuration

**File**: `config/data_optimization_config.yaml`
```yaml
data_sources:
  user_feedback:
    weight: 1.0
    min_samples: 10
  scraped_data:
    weight: 0.7
    min_quality_score: 0.6
    enabled: true
  synthetic_data:
    weight: 0.3
    enabled: true

training:
  curriculum_learning: true
  weighted_sampling: true
  domain_fine_tuning: true
  
evaluation:
  test_set_size: 0.15
  validation_set_size: 0.15
  metrics: ["precision@1", "precision@5", "mrr", "ndcg@10"]
```

## Execution Plan

### Week 1: Data Processing
- [ ] Implement data validation pipeline
- [ ] Implement schema mapping
- [ ] Process initial batch of scraped data (5,000 records)
- [ ] Validate data quality

### Week 2: Dataset Creation
- [ ] Create training dataset from processed data
- [ ] Implement data splitting
- [ ] Generate synthetic data
- [ ] Enrich data with ISTA database links

### Week 3: Model Training
- [ ] Fine-tune fault code encoder
- [ ] Enhance OBD encoder
- [ ] Train multi-modal encoder on combined dataset
- [ ] Evaluate on validation set

### Week 4: Retrieval Optimization
- [ ] Implement learned scoring weights
- [ ] Enhance reranker
- [ ] Optimize retrieval pipeline
- [ ] Test on held-out test set

### Week 5: Evaluation
- [ ] Run comprehensive evaluation
- [ ] Compare before/after metrics
- [ ] Perform error analysis
- [ ] Document improvements

### Week 6: Integration & Deployment
- [ ] Integrate optimized models into production
- [ ] Set up continuous improvement pipeline
- [ ] Monitor performance
- [ ] Collect feedback for next iteration

## Success Metrics

### Quantitative Metrics
- **Precision@1**: Increase from baseline to >0.70
- **Precision@5**: Increase from baseline to >0.85
- **MRR**: Increase from baseline to >0.80
- **NDCG@10**: Increase from baseline to >0.90

### Qualitative Metrics
- Reduced ambiguity in recommendations
- Better handling of code combinations
- Improved accuracy for BMW-specific codes
- Better performance on cases with OBD data

## Risk Mitigation

### Data Quality Risks
- **Risk**: Low-quality scraped data
- **Mitigation**: Strict validation, quality scoring, manual review sample

### Overfitting Risks
- **Risk**: Model overfits to scraped data patterns
- **Mitigation**: Regularization, validation monitoring, diverse data sources

### Integration Risks
- **Risk**: Breaking existing functionality
- **Mitigation**: Comprehensive testing, gradual rollout, rollback plan

## Monitoring & Maintenance

### Continuous Monitoring
1. Track model performance metrics weekly
2. Monitor user feedback and ratings
3. Identify new fault codes or patterns
4. Update training data quarterly

### Model Retraining
- Retrain when:
  - New scraped data available (>1,000 new records)
  - Performance degrades
  - New fault codes emerge
  - User feedback indicates issues

## Conclusion

This plan provides a comprehensive approach to using scraped web data to optimize MIST's accuracy. By following this phased approach, we can systematically improve the system while maintaining reliability and avoiding regressions.

# MIST Data Optimization Summary

## Overview

This document summarizes the complete data optimization strategy for improving MIST's accuracy through web-scraped training data.

## Documents & Scripts

1. **[WEB_SCRAPING_PROMPT.md](../scraping/WEB_SCRAPING_PROMPT.md)** – Prompt for web scraping agents
2. **scripts/process_scraped_data.py** – Data processing pipeline
3. **[TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md)** – Plan to use scraped data (Neon) for training

## Quick Start

### Step 1: Extract Valid Titles
First, generate a list of valid repair guide titles for the scraping agent to use for matching:

```bash
python scripts/extract_repair_guide_titles.py --format agent
```

This creates `data/training/valid_repair_guide_titles.json` with all valid titles from the MIST database.

### Step 2: Collect Data
Use the prompt in [WEB_SCRAPING_PROMPT.md](../scraping/WEB_SCRAPING_PROMPT.md) with a web scraping agent to collect automotive diagnostic data from:
- Forums (Reddit, Bimmerforums, etc.)
- Documentation sites (OBD-Codes.com, AutoZone, etc.)
- Video content (YouTube, etc.)
- Technical Service Bulletins

**Target**: 5,000-10,000 high-quality records

### Step 3: Process Data
Run the data processing script:

```bash
# Process a single file
python scripts/process_scraped_data.py \
    data/scraped/raw_data/forums/reddit_*.jsonl \
    data/scraped/processed_data/reddit_processed.jsonl \
    --min-quality 0.6

# Process entire directory
python scripts/process_scraped_data.py \
    data/scraped/raw_data/ \
    data/scraped/processed_data/
```

### Step 4: Connect to Training

See [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md) for:
- Loading scraped data from Neon/Postgres
- Extending EmbeddingTrainer for scraped pairs
- Guide matching and linking to ISTA procedures

## Expected Improvements

### Current State
- Limited training data (relies on user feedback, minimum 10-100 samples)
- Pre-trained embeddings without domain fine-tuning
- Accuracy issues reported by users

### After Optimization
- **Precision@1**: Target >0.70 (from baseline)
- **Precision@5**: Target >0.85 (from baseline)
- **MRR**: Target >0.80 (from baseline)
- **NDCG@10**: Target >0.90 (from baseline)

## Key Features

### Data Collection
- Comprehensive web scraping strategy
- Multiple data sources (forums, docs, videos, TSBs)
- Quality scoring and validation
- Ethical scraping practices

### Data Processing
- Automatic validation and normalization
- OBD-II data standardization
- Repair procedure extraction
- Deduplication
- Quality scoring

### Training Enhancements
- Multi-source data loading
- Weighted sampling (user feedback > scraped > synthetic)
- Curriculum learning
- Domain-specific fine-tuning
- Enhanced OBD encoder

### Retrieval Optimization
- Learned scoring weights
- Fault code category-specific retrieval
- Enhanced reranking
- Vehicle context awareness

## Next Steps

1. Start web scraping using [WEB_SCRAPING_PROMPT.md](../scraping/WEB_SCRAPING_PROMPT.md)
2. Process data with `process_scraped_data.py`
3. Implement training enhancements per [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md)

## Files Reference

### Documentation
- [WEB_SCRAPING_PROMPT.md](../scraping/WEB_SCRAPING_PROMPT.md) – Web scraping instructions
- [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md) – Training pipeline improvements

### Scripts
- `scripts/extract_repair_guide_titles.py` - Extract valid titles for scraping agent matching
- `scripts/process_scraped_data.py` - Data processing pipeline
- (To be created) `scripts/map_scraped_to_mist_schema.py`
- (To be created) `scripts/create_training_dataset.py`
- (To be created) `scripts/fine_tune_fault_code_encoder.py`
- (To be created) `scripts/evaluate_model.py`

### Configuration
- (To be created) `config/data_optimization_config.yaml`

## Support

- [WEB_SCRAPING_PROMPT.md](../scraping/WEB_SCRAPING_PROMPT.md) – Data collection guidelines
- [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md) – Training pipeline details

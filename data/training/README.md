# Training Data Directory

This directory contains training data and reference files for the MIST system.

## Valid Repair Guide Titles CSV

**File**: `valid_repair_guide_titles.csv`

This CSV file contains all valid repair guide titles and descriptions extracted from the MIST database and vector store. It is used by web scraping agents to match scraped content against known valid repair guides.

### Generating the CSV

To generate or regenerate this file, run:

```bash
# Basic: Extract titles only
python scripts/extract_repair_guide_titles.py --format csv --output data/training/valid_repair_guide_titles.csv

# With descriptions from vector store (default)
python scripts/extract_repair_guide_titles.py --format csv --include-descriptions --output data/training/valid_repair_guide_titles.csv

# With descriptions from XML database (recommended for better coverage)
python scripts/extract_repair_guide_titles.py --format csv --include-descriptions --use-xml-db --output data/training/valid_repair_guide_titles.csv

# With LLM-generated descriptions for missing entries (requires API keys)
# Using Gemini (default, uses GEMINI_API_KEY and GEMINI_MODEL from .env)
python scripts/extract_repair_guide_titles.py --format csv --include-descriptions --use-xml-db --use-llm --llm-provider gemini --output data/training/valid_repair_guide_titles.csv

# Or use OpenAI
python scripts/extract_repair_guide_titles.py --format csv --include-descriptions --use-xml-db --use-llm --llm-provider openai --output data/training/valid_repair_guide_titles.csv
```

**Description Sources (in priority order):**
1. **Vector Store**: Descriptions from indexed repair guides (fastest, but may not cover all procedures)
2. **XML Database**: Extracts content from `xmlvalueprimitive_ENGB.sqlite` (better coverage, slower)
3. **LLM Generation**: Uses AI to generate descriptions for entries missing descriptions (best coverage, requires API keys and costs money)

**Recommendation**: Use `--use-xml-db` for best coverage. Add `--use-llm` only if you need descriptions for entries that aren't in the XML database.

### CSV Format

The CSV contains the following columns:
- `title`: Repair guide title
- `procedure_id`: Unique procedure identifier
- `description`: Summary/description of the repair procedure (first ~500 chars)
- `source`: Source of the data (`ista_db` or `vector_store`)

### Usage

This file is referenced by the web scraping prompt (`docs/WEB_SCRAPING_PROMPT.md`) for matching scraped repair descriptions to valid MIST repair guide titles.

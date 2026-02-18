# Web Scraper Architecture for MIST Training Data

## Overview

This document outlines the architecture for the web scraping system that collects automotive diagnostic training data for MIST.

## Decision: Build in Existing Project

**Why not a separate project?**
- ✅ Already have `process_scraped_data.py` that expects scraped data
- ✅ Can reuse existing utilities (paths, config, logging)
- ✅ Single codebase to maintain
- ✅ Direct integration with MIST data pipeline

**Why not pure open-source solution?**
- ❌ No single tool handles all sources (forums, docs, videos, TSBs)
- ❌ Need custom data extraction logic for automotive-specific content
- ❌ Need integration with MIST data format
- ✅ Hybrid approach: Use open-source libraries + custom scrapers

## Architecture

```
scripts/
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py          # Base class with common functionality
│   ├── forum_scraper.py          # Reddit, forums (Bimmerforums, etc.)
│   ├── documentation_scraper.py # AutoZone, OBD-Codes.com, etc.
│   ├── video_scraper.py          # YouTube, Vimeo
│   ├── tsb_scraper.py            # NHTSA TSB database
│   ├── obd_code_scraper.py       # OBD code databases
│   └── utils/
│       ├── __init__.py
│       ├── extractors.py         # Data extraction utilities
│       ├── validators.py         # Data validation
│       └── rate_limiter.py       # Rate limiting & robots.txt
│
├── scrape_all.py                 # Main orchestrator script
└── process_scraped_data.py       # Existing processor (already exists)
```

## Technology Stack

### Core Libraries

1. **BeautifulSoup4** + **httpx/requests**
   - For simple HTML scraping (documentation sites, forums)
   - Lightweight, fast, Pythonic

2. **Scrapy** (optional, for large-scale scraping)
   - Powerful framework for complex sites
   - Built-in rate limiting, retries, pipelines
   - Use if scraping >10,000 pages from single domain

3. **Selenium/Playwright** (for JavaScript-heavy sites)
   - Only if site requires JavaScript rendering
   - More resource-intensive, use sparingly

4. **PRAW** (Python Reddit API Wrapper)
   - Official Reddit API access
   - Better than scraping Reddit HTML

5. **yt-dlp** (YouTube downloader/extractor)
   - Extract video metadata, transcripts, comments
   - More reliable than YouTube API for scraping

6. **langchain** (already in requirements)
   - Can use for text extraction and summarization

### Data Storage

- **JSONL format** (one JSON object per line)
- Output to `data/training/raw_data/` directory structure
- Separate files by source type (forums/, documentation/, videos/, etc.)

## Implementation Strategy

### Phase 1: Base Infrastructure (Week 1)

1. Create base scraper class with:
   - Rate limiting (not respect robots.txt as large companies don't do this and we are using this data for ethical means)
   - Retry logic
   - Error handling
   - Progress tracking
   - Checkpointing (resume capability)

2. Data extraction utilities:
   - Fault code extraction (regex patterns)
   - OBD data extraction from text
   - Vehicle context extraction
   - Repair summary extraction

3. Validation utilities:
   - Fault code validation
   - OBD data validation
   - Quality scoring

### Phase 2: Source-Specific Scrapers (Week 2-3)

1. **Forum Scraper** (highest priority)
   - Reddit (PRAW API)
   - Bimmerforums, E90Post (BeautifulSoup)
   - Stack Exchange (API or scraping)

2. **Documentation Scraper**
   - AutoZone Repair Guides
   - OBD-Codes.com
   - CarParts.com

3. **Video Scraper**
   - YouTube (yt-dlp for metadata/transcripts)
   - Extract fault codes from titles/descriptions

4. **TSB Scraper**
   - NHTSA TSB database (API or scraping)

5. **OBD Code Scraper**
   - OBD-Codes.com
   - Engine-Codes.com

### Phase 3: Orchestration & Quality (Week 4)

1. Main orchestrator script (`scrape_all.py`)
   - Coordinate all scrapers
   - Progress reporting
   - Data aggregation

2. Quality checks
   - Deduplication
   - Quality scoring
   - Validation

3. Integration with `process_scraped_data.py`

## Key Design Decisions

### 1. Rate Limiting
- Default: 1 request per 2 seconds
- Respect robots.txt
- Configurable per source
- Exponential backoff on errors

### 2. Checkpointing
- Save progress every N records or M minutes
- Resume from checkpoint on restart
- Prevents data loss on crashes

### 3. Error Handling
- Log all errors but continue
- Retry transient errors (429, 503)
- Skip problematic pages, don't fail entire run

### 4. Data Format
- Output JSONL immediately (streaming)
- No need to hold all data in memory
- Easy to process incrementally

### 5. Modularity
- Each scraper is independent
- Can run individually or together
- Easy to add new sources

## Example Usage

```bash
# Scrape all sources
python scripts/scrape_all.py --output data/training/raw_data

# Scrape only forums
python scripts/scrapers/forum_scraper.py --output data/training/raw_data/forums

# Scrape with specific configuration
python scripts/scrape_all.py \
    --sources forums,documentation \
    --max-pages 1000 \
    --rate-limit 2.0 \
    --output data/training/raw_data

# Resume from checkpoint
python scripts/scrape_all.py --resume data/training/raw_data/.checkpoint.json
```

## Dependencies to Add

```txt
# Web Scraping
beautifulsoup4>=4.12.0
lxml>=4.9.0  # Already in requirements
praw>=7.7.0  # Reddit API
yt-dlp>=2023.10.0  # YouTube extraction
selenium>=4.15.0  # Optional, for JS-heavy sites
playwright>=1.40.0  # Optional, alternative to Selenium
scrapy>=2.11.0  # Optional, for large-scale scraping
```

## Ethical Considerations

1. **Respect robots.txt** - Always check before scraping
2. **Rate limiting** - Don't overload servers
3. **User-Agent** - Identify as a research bot
4. **Terms of Service** - Review and comply with ToS
5. **Data privacy** - Anonymize usernames, remove PII
6. **Attribution** - Include source URLs

## Next Steps

1. ✅ Create architecture document (this file)
2. ⏳ Add scraping dependencies to requirements.txt
3. ⏳ Create base scraper class
4. ⏳ Implement forum scraper (start with Reddit)
5. ⏳ Test with small dataset
6. ⏳ Expand to other sources
7. ⏳ Integrate with existing processing pipeline

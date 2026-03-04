# MIST Scripts

Utility scripts for database migrations, indexing, scraping, training, and maintenance.

**Preferred:** Use `mist-cli` after `pip install -e .` (avoids conflict with npm `mist`):

```bash
mist-cli migrate              # Run migrations
mist-cli readiness            # Check index readiness
mist-cli build-kg            # Build knowledge graph
mist-cli train               # Train embeddings
mist-cli scrape forum        # Run forum spider
mist-cli index               # Index repair guides
mist-cli extract-titles      # Extract repair guide titles
mist-cli fetch-bmwfault      # Fetch P-code mappings from bmwfault.codes (HTTP mode by default)
mist-cli --help              # See all commands
```

The scripts below are thin wrappers that call into `src.commands` or the `mist` CLI.

## Migrations

| Script | Purpose |
|--------|---------|
| `run_migrations.py` | Create/update MIST SQLite database (`mist_data.db`) |
| `run_postgres_migration.py` | Migrate `scraped_records` table to Postgres |
| `run_indexing_work_migration.py` | Create `indexing_work` table for multi-machine indexing |

## Indexing

| Script | Purpose |
|--------|---------|
| `index_repair_guides.py` | Index repair procedures from ISTA DB into ChromaDB vector store |
| `check_index_readiness.py` | Verify prerequisites before indexing (ChromaDB, ISTA DB, configs) |

## Knowledge Graph

| Script | Purpose |
|--------|---------|
| `build_knowledge_graph.py` | Build NetworkX knowledge graph from BMW ISTA database |

## Scraping

| Script | Purpose |
|--------|---------|
| `run_scraper.py` | Run Scrapy spiders (forum, doc, example) |
| `process_scraped_data.py` | Process raw scraped JSONL into training format |
| `export_scraped_from_postgres.py` | Export scraped records from Postgres |

## Training

| Script | Purpose |
|--------|---------|
| `train_embeddings.py` | Fine-tune embeddings from feedback data (contrastive learning) |

## Feedback & Analysis

| Script | Purpose |
|--------|---------|
| `collect_feedback.py` | Collect and analyze feedback from feedback database |

## Extraction & Matching

| Script | Purpose |
|--------|---------|
| `extract_repair_guide_titles.py` | Extract valid repair guide titles for scraping agents |
| `match_repair_guides.py` | Match scraped content to valid repair guides |
| `verify_embedding_match.py` | Verify embedding/vector store consistency |

## Ranking & Tuning

| Script | Purpose |
|--------|---------|
| `tune_ranking_weights.py` | Tune retrieval ranking weights |

## E2E & Testing

| Script | Purpose |
|--------|---------|
| `e2e_match_test.py` | End-to-end match test |
| `e2e_one_procedure_one_record.py` | E2E test: one procedure, one record |

## BMW Fault Code Mappings

| Script | Purpose |
|--------|---------|
| `fetch_bmwfault_mappings.py` | Fetch P-code → BMW hex mappings and full row data from bmwfault.codes |

Data is stored in `mist_data.db` (run `mist-cli migrate` first):
- **bmwfault_pcodes**: Full rows (PCode, Code, Label, ECU Variant, ECU Label, fault_info JSON from DiagView)
- **bmwfault_mappings**: Aggregated pcode → hex_codes for `get_lookup_variants()`

**HTTP mode:** Set `CLOUDFLARE_COOKIES` in `.env` or `data/bmwfault_cookies.txt`. `CAPSOLVER_API_KEY` for Turnstile. Use `--refresh-cookies` to refresh via HTTP+Capsolver (works on WSL).

**IP block / VPN:** If you see "This service is no longer available to you", the script waits 10 min (for VPN IP rotation) then retries. Proxies (`BMWFAULT_PROXY`) are optional.

**Rate limiting:** On app-level rate limit ("too often") or Capsolver timeout: exponential backoff (9→27→81 min). Use `-v` to save debug HTML when empty responses contain error alerts.

**Options:** `--no-diagview` skips fetching DiagView pages (faster, no fault_info JSON).

## Other

| Script | Purpose |
|--------|---------|
| `update_fault_codes.py` | Update fault codes in database |
| `run_chroma_mcp.py` | Launch ChromaDB MCP server (loads `.env`) |

## Running Scripts

Ensure the project root is on `PYTHONPATH` (or run `pip install -e .`):

```bash
# From project root
PYTHONPATH=. python scripts/run_migrations.py

# Or after pip install -e .
python scripts/run_migrations.py
```

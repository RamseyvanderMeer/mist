# MIST Documentation Index for AI Agents

> Use `Read` with the paths below to load relevant docs.

## Quick Navigation

| Need | Doc |
|------|-----|
| Cursor/IDE setup, run, test, lint hooks | [AGENTS.md](../AGENTS.md) (`lefthook.yml`, `pip install -e ".[dev]"`) |
| **Repo spec (paths, flows, stack) — read first for code work** | [SPEC.md](SPEC.md) |
| **AI placement, safety, conventions** | [REPO_RULES_FOR_AI.md](REPO_RULES_FOR_AI.md) |
| Project overview & setup | [README.md](../README.md) |
| Architecture & getting started | [ARCHITECTURE.md](ARCHITECTURE.md) |
| BMW ISTA database | [DATABASE.md](DATABASE.md), [ISTA_DATABASE_GUIDE.md](ISTA_DATABASE_GUIDE.md) |
| Web scraping | [SCRAPER_ARCHITECTURE.md](SCRAPER_ARCHITECTURE.md), [WEB_SCRAPING_PROMPT.md](WEB_SCRAPING_PROMPT.md) |
| Training pipeline (scraped data) | [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md), [DATA_OPTIMIZATION_SUMMARY.md](DATA_OPTIMIZATION_SUMMARY.md) |
| Project restructuring | [RESTRUCTURING_PROPOSAL.md](RESTRUCTURING_PROPOSAL.md) |
| CLI usage | Use `mist-cli migrate`, `mist-cli train`, etc. after `pip install -e .` |
| Retrieval evaluation | `tests/test_retrieval_evaluation.py` — eval using scraped_records (matched_guide_id); requires DATABASE_URL, ChromaDB, ISTA |
| Matcher accuracy | `tests/test_matcher_accuracy.py` — compare retrieved guides to solution; default symptom-only query, optional `MATCHER_EVAL_QUERY_MODE=solution` or `--query-mode solution` benchmark |
| Matcher accuracy delta | `tests/test_matcher_accuracy.py::test_matcher_accuracy_query_mode_delta` — runs symptom and solution modes and prints delta |
| Retrieval mismatch | [RETRIEVAL_MISMATCH_INVESTIGATION.md](RETRIEVAL_MISMATCH_INVESTIGATION.md) — why similarity ≠ LLM match, and recent fixes (title fallback, procedure dedup, min_similarity) |

## docs/ Contents

| File | Description |
|------|-------------|
| [SPEC.md](SPEC.md) | Concise path-grounded spec: stack, structure, flows, gotchas |
| [REPO_RULES_FOR_AI.md](REPO_RULES_FOR_AI.md) | Rules for AI: file placement, testing, safety, doc updates |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, components, config, API, getting started |
| [DATABASE.md](DATABASE.md) | BMW ISTA database overview, tables, paths, access |
| [ISTA_DATABASE_GUIDE.md](ISTA_DATABASE_GUIDE.md) | Document hierarchy, Process Analysis, preliminary tasks |
| [SCRAPER_ARCHITECTURE.md](SCRAPER_ARCHITECTURE.md) | Web scraper design, tech stack, folder structure |
| [SCRAPER_DISCOVERY.md](SCRAPER_DISCOVERY.md) | Scraping sources (Bimmerpost, Reddit, etc.) and status |
| [WEB_SCRAPING_PROMPT.md](WEB_SCRAPING_PROMPT.md) | Agent prompt for collecting automotive diagnostic training data |
| [DATA_OPTIMIZATION_SUMMARY.md](DATA_OPTIMIZATION_SUMMARY.md) | Data optimization quick start and next steps |
| [TRAINING_PIPELINE_IMPROVEMENT_PLAN.md](TRAINING_PIPELINE_IMPROVEMENT_PLAN.md) | Plan to use scraped (Neon) data for training |
| [RESTRUCTURING_PROPOSAL.md](RESTRUCTURING_PROPOSAL.md) | Project structure analysis and restructuring recommendations |

## Other Locations

| File | Description |
|------|-------------|
| [data/databases/README.md](../data/databases/README.md) | Database files, primary DB, XML/stream DBs |
| [data/training/README.md](../data/training/README.md) | Training data, valid_repair_guide_titles.csv |
| [scrapers/PRODUCTION_READINESS_ANALYSIS.md](../scrapers/PRODUCTION_READINESS_ANALYSIS.md) | Scraper production gaps and checklist |

## Path Reference

```
e:\mist\AGENTS.md
e:\mist\README.md
e:\mist\docs\SPEC.md
e:\mist\docs\REPO_RULES_FOR_AI.md
e:\mist\docs\ARCHITECTURE.md
e:\mist\docs\DATABASE.md
e:\mist\docs\ISTA_DATABASE_GUIDE.md
e:\mist\docs\SCRAPER_ARCHITECTURE.md
e:\mist\docs\SCRAPER_DISCOVERY.md
e:\mist\docs\WEB_SCRAPING_PROMPT.md
e:\mist\docs\DATA_OPTIMIZATION_SUMMARY.md
e:\mist\docs\TRAINING_PIPELINE_IMPROVEMENT_PLAN.md
e:\mist\data\databases\README.md
e:\mist\data\training\README.md
e:\mist\scrapers\PRODUCTION_READINESS_ANALYSIS.md
```

# MIST Project Restructuring Proposal

This document summarizes a full analysis of the MIST project structure and recommends changes for a more cohesive, maintainable layout.

---

## Executive Summary

The MIST project is **generally well-organized** with a clear separation between core application (`src/`), scrapers (`scrapers/`), scripts, config, and docs. The main improvements are:

1. **Standardize import behavior** across scripts (use project root + `from src.X`)
2. **Add `pyproject.toml`** for proper package discovery and install
3. **Fix broken references** (e.g., `migrate_databases.py`)
4. **Reorganize scripts** into logical subdirectories
5. **Unify test layout** (either flat or clearly categorized)
6. **Consider integrating scrapers** under `src/` for stronger cohesion (optional)

---

## Current Structure Overview

```
mist/
├── src/              # Core application (API, retrieval, embeddings, DB, LLM, feedback)
├── scrapers/         # Scrapy-based web scrapers
├── scripts/          # 18+ flat scripts + migrations/
├── config/           # YAML configs
├── data/             # Databases, training data, vector store, checkpoints
├── docs/             # Documentation
├── tests/            # Mix of root tests, unit/, integration/, e2e/
├── mcps/             # MCP descriptors (Cursor)
├── requirements.txt
├── requirements-scraper.txt
└── scrapy.cfg
```

---

## Issues Identified

### 1. Inconsistent `sys.path` Handling Across Scripts

Scripts use **three different patterns** for imports:

| Pattern | Scripts | Imports | Issue |
|---------|---------|---------|-------|
| Add **project root** | `run_migrations.py` | `from src.database import ...` | ✅ Correct |
| Add **src/** | `index_repair_guides.py`, `check_index_readiness.py`, `train_embeddings.py`, `extract_repair_guide_titles.py` | `from database.X`, `from paths` | ❌ Inconsistent; breaks if run from wrong cwd |
| Add **both** root and src | `tune_ranking_weights.py`, `match_repair_guides.py`, `process_scraped_data.py`, `e2e_one_procedure_one_record.py` | `from src.X` | ✅ Works but redundant |

**Recommendation:** All scripts should add **only the project root** to `sys.path` and use `from src.X` consistently. AGENTS.md already documents `PYTHONPATH=/workspace`; scripts should follow the same convention.

### 2. No Package Installability

- No `pyproject.toml` or `setup.py`
- Reliance on `PYTHONPATH` or `sys.path.insert` in every script
- Can't `pip install -e .` for editable development

**Recommendation:** Add a minimal `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "mist"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []  # Or read from requirements.txt

[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "scrapers*"]
```

Then `pip install -e .` makes `src` and `scrapers` importable without manual path setup.

### 3. Broken Documentation Reference

`data/databases/README.md` (lines 62–64) references:

```bash
python scripts/migrate_databases.py
```

**This file does not exist.** The actual migration script is `scripts/run_migrations.py`.

**Recommendation:** Update `data/databases/README.md` to reference `run_migrations.py` and remove the stale `migrate_databases.py` reference.

### 4. Flat Scripts Directory

There are 18+ scripts in a single `scripts/` directory with mixed purposes:

| Category | Scripts |
|----------|---------|
| **Migrations** | `run_migrations.py`, `run_postgres_migration.py`, `run_indexing_work_migration.py` |
| **Indexing** | `index_repair_guides.py`, `check_index_readiness.py` |
| **Knowledge graph** | `build_knowledge_graph.py` |
| **Scraping** | `run_scraper.py`, `process_scraped_data.py` |
| **Training** | `train_embeddings.py` |
| **Feedback** | `collect_feedback.py` |
| **Extraction/Matching** | `extract_repair_guide_titles.py`, `match_repair_guides.py`, `verify_embedding_match.py` |
| **Ranking** | `tune_ranking_weights.py` |
| **E2E/Testing** | `e2e_match_test.py`, `e2e_one_procedure_one_record.py` |
| **Updates** | `update_fault_codes.py` |
| **Export** | `export_scraped_from_postgres.py` |
| **MCP** | `run_chroma_mcp.py` |

**Recommendation:** Group scripts into subdirectories for clarity:

```
scripts/
├── migrations/       # Keep existing SQL files
├── run/
│   ├── migrations.py
│   ├── scraper.py
│   └── chroma_mcp.py
├── indexing/
│   ├── index_repair_guides.py
│   └── check_readiness.py
├── training/
│   └── train_embeddings.py
├── scraping/
│   ├── process_scraped_data.py
│   └── export_scraped_from_postgres.py
└── ...
```

Or use a **lighter touch**: keep scripts flat but add a `scripts/README.md` that categorizes them. Subdirectories would require updating all references (AGENTS.md, README.md, Dockerfile, etc.).

### 5. Test Layout Inconsistency

- ~20 tests at `tests/` root
- `tests/unit/` has a few (e.g., `test_extractors.py`, `test_process_scraped_data.py`)
- `tests/integration/` has `test_doc_spider.py`
- `tests/e2e/` has `test_runner.py` (excluded in cloud per AGENTS.md)

**Recommendation:** Either:
- **Option A:** Move all tests under `unit/` and `integration/` by domain (retrieval, embeddings, api, scrapers, etc.).
- **Option B:** Keep flat structure but add a `tests/README.md` documenting test categories and how to run them.

Option B is lower-risk; Option A improves discoverability but requires moving many files.

### 6. Scrapers vs Core Placement

- **Current:** `scrapers/` is a top-level package alongside `src/`
- **Coupling:** `doc_spider` imports `src.database.ista_db`; `process_scraped_data.py` imports both `src` and `scrapers`
- **SCRAPER_ARCHITECTURE.md** describes an older layout (`scripts/scrapers/`) that doesn't match the current `scrapers/` at root

**Recommendation (conservative):** Keep `scrapers/` at root. Update `SCRAPER_ARCHITECTURE.md` to reflect the current layout. The current separation is reasonable: scrapers are a distinct subsystem that happens to depend on core.

**Recommendation (aggressive):** Move `scrapers/` to `src/scrapers/` for a single `src` package. Would require:
- Updating `scrapy.cfg` to `[settings] default = src.scrapers.settings`
- Updating `run_scraper.py` spider paths
- Updating all `from scrapers.X` to `from src.scrapers.X`

### 7. Paths Class and Environment Variables

`src/paths.py` is well-designed with env overrides (`MIST_DATABASE_DIR`, `MIST_CONFIG_DIR`, etc.). No changes needed.

### 8. Config and Data Layout

- `config/`: YAML files are logically organized
- `data/`: Subdirs `databases/`, `training/` are clear; `Paths` class encapsulates access

No restructuring needed.

---

## Recommended Actions (Prioritized)

### High Priority (Low Risk)

1. **Fix `data/databases/README.md`** – Replace `migrate_databases.py` reference with `run_migrations.py`.
2. **Standardize script imports** – Change scripts that add `src/` to add project root only, and use `from src.X` everywhere. Scripts to update:
   - `scripts/check_index_readiness.py` (currently adds `ROOT / "src"`)
   - `scripts/index_repair_guides.py` (adds `src/`, uses bare imports)
   - `scripts/train_embeddings.py` (adds `src/`)
   - `scripts/extract_repair_guide_titles.py` (adds `src/`)

### Medium Priority

3. **Add `pyproject.toml`** – Enable `pip install -e .` for clean imports.
4. **Add `scripts/README.md`** – Categorize scripts by purpose with one-line descriptions.
5. **Update `docs/SCRAPER_ARCHITECTURE.md`** – Match the document to the actual `scrapers/` layout.

### Lower Priority (More Invasive)

6. **Reorganize scripts** into subdirs – Only if the flat structure becomes hard to navigate.
7. **Reorganize tests** – Move unit/integration tests into clearer hierarchy.
8. **Move scrapers under src** – Only if you want a single top-level source package.

---

## Proposed Target Structure (Minimal Changes)

```
mist/
├── pyproject.toml          # NEW: Package metadata + editable install
├── src/
│   ├── api/
│   ├── database/
│   ├── embeddings/
│   ├── feedback/
│   ├── knowledge/
│   ├── learning/
│   ├── llm/
│   ├── retrieval/
│   └── paths.py
├── scrapers/
├── scripts/
│   ├── README.md           # NEW: Script index
│   └── migrations/
├── config/
├── data/
├── docs/
└── tests/
```

No folder moves; only new files and import/path fixes.

---

## Implementation Checklist

- [x] Update `data/databases/README.md` (remove migrate_databases.py, fix migration command)
- [x] Fix `scripts/check_index_readiness.py` – use project root, `from src.X`
- [x] Fix `scripts/index_repair_guides.py` – use project root, `from src.X`
- [x] Fix `scripts/train_embeddings.py` – use project root, `from src.X`
- [x] Fix `scripts/extract_repair_guide_titles.py` – use project root, `from src.X`
- [x] Add `pyproject.toml` and document `pip install -e .` in README
- [x] Add `scripts/README.md`
- [x] Update `docs/SCRAPER_ARCHITECTURE.md` to match actual layout

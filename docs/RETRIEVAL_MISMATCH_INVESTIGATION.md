# Retrieval Mismatch Investigation

## Problem

Matcher accuracy evaluation shows **100% similarity-based accuracy** but **0% LLM-based accuracy**. The semantic similarity (embedding cosine) marks retrieved guides as matches, but the LLM correctly identifies they describe different repairs (e.g., "Windscreen, motor power consumption" for a TPS replacement, "Brake light" for a CCV hose fix).

## Root Causes

### 1. Symptom-to-fix lexical gap in retrieval query

Current retrieval paths are symptom-first and typically build `description` from
`repair_summary` and `symptoms`. The most discriminative text is often the actual
fix phrase itself (e.g., "replace throttle position sensor"), which may not appear
in `repair_summary`. Vector search can therefore miss the right procedure unless
the symptom phrasing is expanded or inferred.

**Impact:** Queries like `Fault codes: P0121, P0420, P0430. Problem: [symptoms]`
may miss the exact procedure language and return semantically close but incorrect
guides.

### 2. No fault-code metadata filter in vector search

Vector search uses `filter_dict=None`. ChromaDB supports filtering by `fault_codes` metadata. Indexed procedures store fault codes from ISTA (`RG_ECUFAULT_DOCIDS`), so we can restrict results to procedures that explicitly list at least one of our fault codes.

**Impact:** Irrelevant procedures (e.g., windscreen, brake light) can rank highly because the embedding model matches generic automotive language. Filtering by fault codes would keep results fault-code–relevant.

### 3. Chunk-level retrieval vs procedure-level evaluation

The index stores **chunks** (≈1200 chars) of procedures. Retrieval returns individual chunks. A chunk from procedure A can outrank the best chunk from procedure B. For evaluation we take the top chunk, which may come from the wrong procedure.

**Mitigation:** Procedure-level aggregation (e.g., take best chunk per procedure, then rank procedures) would help, but requires broader pipeline changes.

### 4. Fault code format mismatch

ISTA stores codes in `XEP_FAULTCODES` (e.g., `P0301`, `29CC`, `420`). Scraped records use P-codes. `get_search_codes()` maps P-codes to BMW variants. If a P-code has no mapping (e.g., `P1500`, `P1503`), we search with the raw code only—ISTA may use a different format, causing misses.

### 5. Procedures with empty or "-" title

Some ISTA procedures have `TITLE_ENGB = "-"` or empty. ChromaDB stores this as-is. The LLM sees "Guide: -" and has little semantic signal from the title alone.

## Implemented Improvements

1. **Fault-code metadata filter** – Pass `filter_dict={"fault_codes": search_codes}` to ChromaDB search; fall back to unfiltered search if the filter returns no results.
   - Chroma Cloud quotas can reject very large `fault_codes` predicates. `src/retrieval/chroma_store.py` now batches fault-code filters in chunks (default max 8) and merges the results to avoid `NumWherePredicates` quota errors.
   - You can tune the chunk size via `CHROMA_MAX_WHERE_PREDICATES` if your plan allows more predicates.
2. **Symptom expansion for retrieval** – Keep retrieval symptom-driven, but add
   optional symptom expansion via `QueryExpander` so "fault symptom" language is
   preprocessed toward automotive repair terminology.
3. **Solution-query benchmark mode retained** – Preserve a dedicated experiment mode
   that appends solution text to retrieval description for ablation comparison, but
   do not use it in default matching paths.
4. **DB sampling performance fix for matcher/retrieval eval** – In
   `tests/test_matcher_accuracy.py` and `tests/test_retrieval_evaluation.py`, replace
   `ORDER BY RANDOM()` candidate sampling with a capped prefetch + Python shuffle
   to avoid full-table random sorting at startup.
5. **Lazy vector-store initialization for seed-only runs** – `scripts/index_repair_guides.py`
   now initializes the vector store only when storing documents. This allows
   `--seed-only` (or other queue-only DB modes) to run without a live
   ChromaDB connection.
6. **Force reseed support for queue refresh** – `scripts/index_repair_guides.py` now
   supports `--force-reseed` and applies it only with `--retry-failed` and/or
   `--only-placeholder-title` so targeted reprocessing can reset existing
   queue rows to pending when needed.
7. **PostgreSQL queue setup optimization** – `scripts/index_repair_guides.py` now
   uses a fast path (`to_regclass`) to skip migration DDL when `indexing_work`
   already exists, so runs with repeated `--seed-only` no longer spend time
   re-running table/index DDL checks each invocation.
8. **Docs** – Add this investigation doc and link from `docs/agent.md`.

## Further Work and status

- **Broader fault code mapping:** Extend `bmwfault_mappings` and `OBD_TO_BMW` for more P-codes.
- **Index improvements:** Ensure procedure titles are populated where possible; consider indexing fault labels more prominently.
- **Reranker query:** Experiment with solution-focused query text for the cross-encoder.
- **Placeholder title fallback:** Implemented. `scripts/index_repair_guides.py` now normalizes `TITLE_ENGB='-'` to procedure name.
- **Procedure-level aggregation:** Implemented. `src/retrieval/enhanced_retriever.py` now deduplicates chunk results per procedure.
- **min_similarity gate:** Implemented. `src/retrieval/enhanced_retriever.py` applies `retrieval.min_similarity` as a Stage 1 filter.

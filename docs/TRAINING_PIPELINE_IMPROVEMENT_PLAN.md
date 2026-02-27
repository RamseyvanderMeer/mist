# Training Pipeline Improvement Plan

**Created:** 2026-02-23  
**Context:** 11,264 scraped records in Neon DB; training pipeline currently uses only user feedback (SQLite).

---

## 1. Database Snapshot (Neon `scraped_records`)

| Metric | Value |
|--------|-------|
| **Total rows** | 11,264 |
| **Unique URLs** | 11,264 |
| **Table size** | ~9.3 MB (data) + ~4.2 MB (indexes) |
| **Source type** | 100% forum |

### Record Type × Outcome

| record_type | outcome | count |
|-------------|---------|-------|
| fault_code | success | 5,057 |
| fault_code | unknown | 4,361 |
| cause_to_solution | success | 865 |
| fault_code | failure | 780 |
| fault_code | partial | 166 |
| cause_to_solution | partial | 20 |
| cause_to_solution | failure | 15 |

### Quality Metrics

- **Records with fault codes:** 10,364 (92%)
- **Confidence score:** avg 0.78, range 0.5–1.0
- **matched_guide_id:** mostly NULL (scraped content not yet linked to ISTA procedures)

### Schema (relevant columns)

- `fault_codes` (text, JSON array)
- `repair_summary` (text)
- `outcome` (success | failure | partial | unknown)
- `confidence_score`, `heuristic_score`, `llm_confidence`
- `vehicle_context`, `obd_data`, `symptoms`
- `matched_guide_id`, `matched_guide_title`, `match_reasoning` (for future linking)

---

## 2. Current Pipeline Gaps

1. **Scraped data not used for training**  
   - `EmbeddingTrainer` reads only from `FeedbackCollector` (SQLite feedback DB).  
   - `process_scraped_data.py` works on JSONL files, not Postgres.

2. **Different data shapes**  
   - Feedback: `(fault_codes, obd_data) → selected_guide` (procedure_id).  
   - Scraped: `(fault_codes, repair_summary)` (free text, no procedure_id).

3. **No scraped content in vector store**  
   - Only ISTA repair guides are indexed.  
   - Real-world forum solutions are not retrievable.

4. **No guide matching**  
   - `matched_guide_id` is NULL; scraped repair summaries are not linked to ISTA procedures.

---

## 3. Improvement Plan

### Phase 1: Connect Scraped Data to Training (High Impact)

**Goal:** Use scraped `(fault_codes, repair_summary)` pairs for contrastive training.

**Tasks:**

1. **Add `ScrapedDataLoader`**
   - Read from Neon `scraped_records` via `DATABASE_URL`.
   - Filter: `outcome IN ('success', 'partial')`, `confidence_score >= 0.7`, non-empty `fault_codes` and `repair_summary`.
   - Yield `(fault_codes, repair_summary)` pairs.

2. **Extend `EmbeddingTrainer.create_dataset()`**
   - Support a `scraped_data_source` (e.g. `"neon"` or `"postgres"`).
   - For scraped pairs:
     - **Anchor:** `encode(fault_codes)` (optionally with `obd_data`).
     - **Positive:** `encode(repair_summary)` (treat repair_summary as document text).
   - **Negatives:** sample from other scraped `repair_summary` with different fault codes, or from vector store.

3. **Unified training data**
   - Combine feedback sessions and scraped pairs.
   - Option: `--data-source feedback|scraped|both` in `train_embeddings.py`.

**Estimated effort:** 2–3 days.

---

### Phase 2: Process Scraped Data from DB (Medium Impact)

**Goal:** Make `process_scraped_data.py` work with Postgres as input.

**Tasks:**

1. **Add `--from-db` to `process_scraped_data.py`**
   - When set, read from `scraped_records` instead of JSONL.
   - Reuse existing validation, quality scoring, deduplication.

2. **Output formats**
   - JSONL (for backward compatibility).
   - Optional: write processed records to a `processed_scraped_records` table for auditing.

3. **Config**
   - Use `DATABASE_URL` from env; support `--db-url` override.

**Estimated effort:** 1 day.

---

### Phase 3: Index Scraped Repair Summaries (Medium Impact)

**Goal:** Add high-quality scraped solutions to the vector store for retrieval.

**Tasks:**

1. **Script `scripts/index_scraped_repairs.py`**
   - Query `scraped_records` with filters (e.g. `outcome = 'success'`, `confidence_score >= 0.8`).
   - Encode `repair_summary` (and optionally `fault_codes` + `vehicle_context`) as document text.
   - Upsert into ChromaDB with metadata: `source_url`, `fault_codes`, `outcome`, `record_type`.

2. **Collection strategy**
   - Option A: Same collection as ISTA guides, with `source: "scraped"` in payload.
   - Option B: Separate collection `scraped_repairs` for experiments.

3. **Retrieval integration**
   - Update retrieval pipeline to optionally search scraped collection or merge results.

**Estimated effort:** 1–2 days.

---

### Phase 4: Guide Matching (Higher Effort, High Value)

**Goal:** Link scraped `repair_summary` to ISTA `procedure_id` for stronger training signal.

**Tasks:**

1. **Matching pipeline**
   - Use `valid_repair_guide_titles.csv` and/or vector search to find best-matching ISTA procedure.
   - LLM or embedding similarity to score matches.
   - Populate `matched_guide_id`, `matched_guide_title`, `match_reasoning`.

2. **Training impact**
   - When `matched_guide_id` is set, use ISTA procedure embedding as positive (same as feedback).
   - When NULL, fall back to encoding `repair_summary` directly (Phase 1 behavior).

3. **Script**
   - `scripts/match_scraped_to_guides.py` to batch-update `scraped_records`.

**Estimated effort:** 2–3 days.

---

### Phase 5: Data Quality and Monitoring (Ongoing)

**Tasks:**

1. **DB indexes**
   - Add composite index on `(outcome, confidence_score)` for training queries.
   - Add index on `(record_type, outcome)` if filtering by both.

2. **Quality dashboard**
   - Simple script or notebook: row counts, outcome distribution, confidence distribution, fault code coverage.

3. **Deduplication**
   - Run `ScrapedDataProcessor.deduplicate_records()` on DB export periodically.
   - Consider DB-level dedup by `(fault_codes, repair_summary_hash)` if needed.

---

## 4. Recommended Implementation Order

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Connect scraped data to training | High | 2–3 days | High |
| 2. Process scraped data from DB | Medium | 1 day | Medium |
| 3. Index scraped repairs in vector store | Medium | 1–2 days | Medium |
| 4. Guide matching | High | 2–3 days | High |
| 5. Data quality and monitoring | Low | Ongoing | Medium |

**Suggested sequence:** Phase 1 → Phase 2 → Phase 3 → Phase 5. Phase 4 can run in parallel once Phase 1 is stable.

---

## 5. Quick Wins (Can Do Now)

1. **Add DB index for training queries:**
   ```sql
   CREATE INDEX IF NOT EXISTS idx_scraped_records_training
   ON scraped_records (outcome, confidence_score)
   WHERE fault_codes IS NOT NULL AND fault_codes != '[]' AND repair_summary IS NOT NULL;
   ```

2. **Export high-quality scraped data to JSONL** for `process_scraped_data.py`:
   ```sql
   -- Run via Neon MCP or psql, export to data/training/raw_data/scraped_from_db.jsonl
   SELECT json_build_object(
     'source_url', source_url,
     'fault_codes', fault_codes::json,
     'repair_summary', repair_summary,
     'outcome', outcome,
     'record_type', record_type,
     'confidence_score', confidence_score
   ) FROM scraped_records
   WHERE outcome IN ('success', 'partial')
     AND confidence_score >= 0.7
     AND fault_codes IS NOT NULL AND fault_codes != '[]'
     AND repair_summary IS NOT NULL;
   ```

3. **Lower `min_feedback_samples`** in `config/training_config.yaml` if you want to train on feedback alone with fewer samples (e.g. for testing).

---

## 6. Configuration Additions

Suggested additions to `config/training_config.yaml`:

```yaml
# Scraped data training (Phase 1)
scraped_data:
  enabled: true
  min_confidence: 0.7
  outcomes: ["success", "partial"]
  max_records: 5000  # Cap for memory/speed
  negative_sampling: "in_batch"  # or "vector_store"
```

---

## 7. Dependencies

- `DATABASE_URL` (or `NEON_DATABASE_URL`) for Postgres/Neon.
- Existing: `FeedbackCollector`, `EmbeddingTrainer`, `MultiModalEncoder`, `VectorStore`, `process_scraped_data.py`.

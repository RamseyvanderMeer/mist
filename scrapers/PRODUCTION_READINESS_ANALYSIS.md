# Scraper Module: Production Readiness Analysis

In-depth analysis of the MIST scraper module for design gaps, bugs, and production requirements.

---

## 1. Architecture & Design Gaps

### 1.1 Missing Capabilities (per docs/SCRAPER_ARCHITECTURE.md)

| Capability | Status | Notes |
|------------|--------|--------|
| **Checkpointing / resume** | Missing | No `.checkpoint.json` or resume-from-last state. Long runs lose progress on crash. |
| **Exponential backoff** | Partial | Scrapy's RetryMiddleware retries with fixed delay; no custom backoff on 429/503. |
| **Configurable rate limit per source** | Partial | `DOWNLOAD_DELAY` is global (2s); forum/doc cannot have different delays. |
| **scrape_all orchestrator** | Missing | Architecture describes `scripts/scrape_all.py --sources forums,documentation`; only `run_scraper.py --spider` exists (one spider per run). |
| **max-pages / CLOSESPIDER** | Missing | No way to cap pages or requests for testing or cost control. |
| **Video spider** | Missing | Plan listed `video_spider.py` (YouTube/yt-dlp); not implemented. |
| **TSB spider** | Missing | NHTSA TSB scraper not implemented. |
| **Reddit via PRAW** | Missing | Forum spider scrapes HTML only; no PRAW integration for API-based Reddit. |

### 1.2 Duplicated Logic

- **Fault code patterns and OBD ranges** exist in three places:
  - `scrapers/utils/extractors.py` (extraction patterns)
  - `scrapers/pipelines/validation.py` (validation patterns)
  - `scripts/process_scraped_data.py` (processing)
- **Risk**: Changes to code formats (e.g. new P-code pattern) must be updated in three files. Should be a single shared module (e.g. `scrapers/utils/schemas.py` or shared constants).

### 1.3 Pipeline Order and Item Types

- **Example spider** yields `dict` with `url`, `html`, `raw_text` (no `fault_codes`). Validation and JsonlWriter correctly pass it through and do not write it to MIST JSONL.
- **LangGraph pipeline** passes items without `html`/`raw_text` (e.g. MistScrapedItem). If LangGraph is later implemented to call an agent, it will need to branch on item type so only raw-content items are sent to the agent.

---

## 2. Bugs & Correctness

### 2.1 Confirmed Bugs

1. **`datetime.utcnow()` deprecated (Python 3.12+)**
   - **Where**: `scrapers/spiders/base.py` line 80.
   - **Fix**: Use `datetime.now(timezone.utc).isoformat()` (or `datetime.now(datetime.UTC)` on 3.11+).
   - **Also**: `scripts/process_scraped_data.py` line 330 has the same issue (separate module but same fix).

2. **Validation pipeline: OBD values in range not stored**
   - **Where**: `scrapers/pipelines/validation.py` in `normalize_obd_data`.
   - **Code**: When `key_lower in OBD_RANGES` and value is in range, `normalized[key_lower] = float_value` is set. When value is *out* of range, the key is skipped (intended). When `key_lower` is *not* in `OBD_RANGES`, the code falls through to `else: normalized[key_lower] = float_value`. So in-range known keys are stored; out-of-range known keys are dropped; unknown keys are stored. **No bug** (re-verified).

3. **Doc spider: no HTTP error handling**
   - **Where**: `scrapers/spiders/doc_spider.py`.
   - **Issue**: 404 or 500 from obd-codes.com are not handled; Scrapy will retry 500s, but 404s are parsed as valid pages and may yield empty or wrong extractions.
   - **Fix**: Use `handle_httpstatus_list = [404]` and in `parse_code_page` check `if response.status == 404: return`, or use `errback` to log and skip.

4. **Forum spider: fragile selectors**
   - **Where**: `scrapers/spiders/forum_spider.py`.
   - **Issue**: Selectors like `.post`, `.message`, `article[data-testid='post-container']` are site-specific. Bimmerforums/E90Post may use different class names; spider may yield no items on real pages.
   - **Fix**: Add site-specific selector mapping (e.g. by domain) or document required structure; consider optional fallback that uses `body` text if no post blocks found.

### 2.2 Potential Bugs

5. **JsonlWriterPipeline: Scrapy Item → dict**
   - **Where**: `scrapers/pipelines/io.py`: `record = dict(item)`.
   - **Issue**: Scrapy Item can have fields that are unset (missing). `dict(item)` may include keys with `None` or leave keys out depending on Scrapy version. `json.dumps` handles None; downstream `process_scraped_data.py` uses `.get()` so missing keys are OK.
   - **Verdict**: Likely safe; worth a quick test with an item that has optional fields missing.

6. **run_scraper exit code**
   - **Where**: `scripts/run_scraper.py` always calls `sys.exit(0)`.
   - **Issue**: If the crawl fails (exception or spider error), process may still exit 0, breaking CI or job orchestration.
   - **Fix**: Use CrawlerProcess’s return value / reactor result or wrap in try/except and `sys.exit(1)` on failure.

---

## 3. Robustness & Production Hardening

### 3.1 Error Handling

- **Spiders**: No `errback` on requests; no explicit handling of non-200 or timeouts. Add `errback` to log and optionally retry or skip URL.
- **Pipelines**: Validation raises `DropItem` (correct). JsonlWriter does not catch `IOError`/`OSError`; a full disk or permission error could crash the spider run. Wrap write in try/except and log; optionally re-raise after logging.
- **run_scraper**: No try/except around `process.crawl` / `process.start()`; any exception propagates and exit code is still 0 if the script doesn’t crash.

### 3.2 Configuration

- **Paths**: `MIST_RAW_DATA_DIR` defaults to `data/training/raw_data` (relative). In production (e.g. Cloud Run), working directory may differ; use absolute path or explicit env (e.g. `MIST_RAW_DATA_DIR`).
- **Secrets**: No API keys in scraper code (good). Reddit/PRAW would require `REDDIT_CLIENT_ID` etc. in env when added.
- **Feature flags**: No way to disable pipelines (e.g. turn off JSONL write for dry-run). Could use Scrapy settings or env.

### 3.3 Observability

- **Logging**: Uses Python logging and Scrapy’s log. No structured logs (JSON) or request/response logging for debugging.
- **Metrics**: No counters for items scraped, items dropped, per-spider stats exposed to a metrics system.
- **Alerting**: No integration with error tracking or alerts; failures are only in logs.

### 3.4 Security & Ethics

- **robots.txt**: `ROBOTSTXT_OBEY = False`; architecture notes this is intentional. Document reason and ensure it’s acceptable for each target domain.
- **User-Agent**: Set to `MIST-Scraper/1.0 (...)`; good for identification.
- **Rate limiting**: Global 2s delay; good. Consider per-domain settings for politeness.
- **PII**: No explicit stripping of names/emails in extracted text; repair summaries might contain usernames or personal details. Consider a post-processing or pipeline step to redact PII before writing JSONL.

---

## 4. Data Quality & Schema

### 4.1 process_scraped_data.py compatibility

- **repair_guide**: JsonlWriterPipeline adds `repair_guide = repair_summary` when writing, so downstream processor receives a string and converts to `{title, procedure_steps}`. Compatible.
- **repair_summary-only input**: If a JSONL file has `repair_summary` but no `repair_guide`, `process_scraped_data.py` does not use `repair_summary`. So external or legacy files with only `repair_summary` would get no repair content. Consider in process_scraped_data: `repair_guide = record.get('repair_guide') or record.get('repair_summary')` (and then same string→dict handling).

### 4.2 Schema drift

- **MistScrapedItem** and `docs/WEB_SCRAPING_PROMPT.md` are aligned. Optional fields are flexible; adding new optional fields in the item is backward compatible as long as pipelines use `.get()`.

---

## 5. Testing

- **No tests found** for `scrapers/` or `scripts/run_scraper.py`, `scripts/process_scraped_data.py`.
- **Recommendations**:
  - Unit tests for `extractors` (fault codes, OBD, vehicle context, outcome) with fixed strings.
  - Unit tests for `validate_fault_code`, `normalize_obd_data`, `calculate_quality_score` in validation pipeline.
  - Integration test: run doc spider with `CLOSESPIDER_PAGECOUNT=1`, assert one JSONL line and required keys.
  - Optional: contract test that one record from JsonlWriterPipeline is accepted by `process_scraped_data.process_record()`.

---

## 6. Production Checklist (Summary)

| Area | Action |
|------|--------|
| **Bugs** | Fix `datetime.utcnow()` in base spider (and process_scraped_data). Add HTTP error handling (404/5xx) and optional errback in doc (and forum) spider. Fix run_scraper exit code on failure. |
| **Design** | Introduce shared constants/schemas for fault codes and OBD ranges. Add checkpointing and resume, or document “no resume” and recommend short runs. Add `--max-pages` or CLOSESPIDER to run_scraper. |
| **Pipelines** | Harden JsonlWriter: catch IOError, log, and decide whether to re-raise. Consider making pipeline order and pipeline enablement configurable. |
| **Config** | Use absolute or env-based path for `MIST_RAW_DATA_DIR` in production. Document env vars for future PRAW/YouTube. |
| **Observability** | Add structured logging and/or metrics (items scraped/dropped, per spider). Optionally integrate with error tracking. |
| **Data** | In process_scraped_data, support `repair_summary` when `repair_guide` is missing. Consider PII redaction before writing JSONL. |
| **Testing** | Add unit tests for extractors and validation; add at least one integration test for doc spider + pipeline + JSONL. |
| **Docs** | Document ROBOTSTXT_OBEY decision, rate limits, and how to run forum vs doc vs example in README or docs/SCRAPER_ARCHITECTURE.md. |

---

## 7. Priority Order for Production

1. **P0 – Must fix**: datetime deprecation; run_scraper exit code on failure; HTTP error handling in doc spider; optional errback for forum.
2. **P1 – Should have**: Shared fault-code/OBD constants; JsonlWriter IOError handling; `repair_summary` fallback in process_scraped_data; basic unit tests for extractors and validation.
3. **P2 – Nice to have**: Checkpointing/resume; `--max-pages`; scrape_all orchestrator; video/TSB spiders; PII redaction; structured logging/metrics.

This document should be updated as gaps are closed and new capabilities (e.g. LangGraph agent, PRAW) are added.

"""
Retrieval evaluation using Neon DB scraped_records table.

Samples records with matched_guide_id (ground truth), runs EnhancedRetriever,
and computes Hit@1, Hit@5, Hit@10, MRR. Optionally writes results to retrieved_records.

Requires: DATABASE_URL (postgresql), CHROMA_DB_*, ISTA DB, knowledge graph.

NOTE: Uses E5-Mistral-7B + cross-encoder — slow on CPU (~1-3 min/record).
Use RETRIEVAL_EVAL_SAMPLE_SIZE=2 for a quick 2-record test.
Use RETRIEVAL_EVAL_DB_CANDIDATE_POOL (default 5000) to control DB prefetch size.

Run:
  PYTHONPATH=. python -m pytest tests/test_retrieval_evaluation.py -v -s
  RETRIEVAL_EVAL_SAMPLE_SIZE=2 python -m pytest tests/test_retrieval_evaluation.py -v -s
  PYTHONPATH=. python tests/test_retrieval_evaluation.py --sample-size 2 --persist
"""
from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse_fault_codes(fc: Any) -> List[str]:
    """Parse fault_codes from JSONB or list."""
    if isinstance(fc, str):
        try:
            fc = json.loads(fc) if fc else []
        except json.JSONDecodeError:
            fc = []
    return [str(c).strip() for c in (fc or []) if c and str(c).strip()]


def _fetch_eval_records(
    db_url: str,
    sample_size: int = 20,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch records from scraped_records with matched_guide_id (ground truth).
    Randomly samples up to sample_size records.

    Uses a capped candidate window instead of ORDER BY RANDOM() for better startup performance.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    try:
        candidate_limit = int(os.environ.get("RETRIEVAL_EVAL_DB_CANDIDATE_POOL", "5000"))
    except ValueError:
        candidate_limit = 5000
    if candidate_limit <= 0:
        candidate_limit = 5000
    if candidate_limit < sample_size:
        candidate_limit = sample_size * 4
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, source_url, fault_codes, repair_summary, symptoms,
                       matched_guide_id, matched_guide_title
                FROM scraped_records
                WHERE matched_guide_id IS NOT NULL AND matched_guide_id != ''
                  AND (repair_summary IS NOT NULL AND repair_summary != ''
                       OR symptoms IS NOT NULL AND symptoms != '')
                  AND fault_codes IS NOT NULL AND fault_codes != '[]' AND fault_codes != '{}'
                LIMIT :candidate_limit
            """),
            {"candidate_limit": candidate_limit},
        )
        rows = list(result.fetchall())
        cols = result.keys()

    records = []
    for row in rows:
        rec = dict(zip(cols, row))
        rec["fault_codes"] = _parse_fault_codes(rec.get("fault_codes"))
        if not rec["fault_codes"]:
            continue
        records.append(rec)

    if seed is not None:
        random.seed(seed)
    random.shuffle(records)
    return records[:sample_size]


def _create_retriever():
    """Create EnhancedRetriever once (model load is slow)."""
    from src.retrieval.enhanced_retriever import EnhancedRetriever

    return EnhancedRetriever()


def _run_retrieval(
    retriever: Any,
    fault_codes: List[str],
    description: Optional[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Run retrieval using shared retriever instance."""
    from src.retrieval.enhanced_retriever import EnhancedRetrieverError

    try:
        results = retriever.retrieve(
            fault_codes=fault_codes,
            obd_data={},
            description=description,
            top_k=top_k,
        )
        return results
    except EnhancedRetrieverError as e:
        raise pytest.skip(f"Retrieval failed (ChromaDB/ISTA required): {e}") from e


def _compute_metrics(
    expected_guide_id: str,
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Hit@1, Hit@5, Hit@10, MRR."""
    seen: set[str] = set()
    retrieved_ids: List[str] = []
    for r in retrieved:
        pid = str(r.get("procedure_id") or r.get("id", "")).strip()
        # Normalize chunk IDs to base procedure_id (e.g. "123_chunk_0" -> "123")
        if "_chunk_" in pid:
            pid = pid.split("_chunk_")[0]
        if pid and pid not in seen:
            seen.add(pid)
            retrieved_ids.append(pid)
    expected = str(expected_guide_id).strip()

    hit_at_1 = expected in retrieved_ids[:1] if retrieved_ids else False
    hit_at_5 = expected in retrieved_ids[:5] if retrieved_ids else False
    hit_at_10 = expected in retrieved_ids[:10] if retrieved_ids else False

    try:
        rank = retrieved_ids.index(expected) + 1
        reciprocal_rank = 1.0 / rank
    except ValueError:
        reciprocal_rank = 0.0

    return {
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_ids": retrieved_ids[:10],
    }


def _persist_to_retrieved_records(
    db_url: str,
    run_id: str,
    records: List[Dict[str, Any]],
) -> None:
    """Write evaluation results to retrieved_records table."""
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    # Ensure table exists
    migration_file = ROOT / "scripts" / "migrations" / "create_retrieved_records_postgres.sql"
    if migration_file.exists():
        with open(migration_file, "r") as f:
            sql = f.read()
        with engine.connect() as conn:
            for stmt in (s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")):
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    conn.rollback()

    with engine.connect() as conn:
        for rec in records:
            conn.execute(
                text("""
                    INSERT INTO retrieved_records (
                        scraped_record_id, source_url, fault_codes, description,
                        expected_guide_id, expected_guide_title,
                        retrieved_guide_ids, retrieved_scores,
                        hit_at_1, hit_at_5, hit_at_10, reciprocal_rank, run_id
                    ) VALUES (
                        :sid, :url, CAST(:fc AS jsonb), :desc,
                        :exp_id, :exp_title,
                        CAST(:ret_ids AS jsonb), CAST(:scores AS jsonb),
                        :h1, :h5, :h10, :rr, :run_id
                    )
                """),
                {
                    "sid": rec.get("scraped_record_id"),
                    "url": rec.get("source_url"),
                    "fc": json.dumps(rec.get("fault_codes", [])),
                    "desc": rec.get("description"),
                    "exp_id": rec.get("expected_guide_id"),
                    "exp_title": rec.get("expected_guide_title"),
                    "ret_ids": json.dumps(rec.get("retrieved_guide_ids", [])),
                    "scores": json.dumps(rec.get("retrieved_scores", [])),
                    "h1": rec.get("hit_at_1"),
                    "h5": rec.get("hit_at_5"),
                    "h10": rec.get("hit_at_10"),
                    "rr": rec.get("reciprocal_rank"),
                    "run_id": run_id,
                },
            )
        conn.commit()


@pytest.fixture(scope="module")
def db_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "")
    if not url or not url.startswith("postgresql"):
        return None
    return url


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").startswith("postgresql"),
    reason="DATABASE_URL (postgresql) required for retrieval evaluation",
)
def test_retrieval_evaluation_random_sample(db_url: Optional[str]):
    """
    Evaluate retrieval by randomly sampling scraped_records with matched_guide_id.

    Runs EnhancedRetriever for each record and checks if the expected repair guide
    appears in top-1, top-5, top-10. Computes Hit@K and MRR.
    """
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    sample_size = int(os.environ.get("RETRIEVAL_EVAL_SAMPLE_SIZE", "2"))
    persist = os.environ.get("RETRIEVAL_EVAL_PERSIST", "").lower() in ("1", "true", "yes")

    records = _fetch_eval_records(db_url, sample_size=sample_size, seed=42)
    if not records:
        pytest.skip("No scraped_records with matched_guide_id found in DB")

    results: List[Dict[str, Any]] = []
    run_id = str(uuid.uuid4())[:8]

    print(f"Initializing retriever (E5-Mistral-7B + reranker)...", flush=True)
    print("  First run may download models (~14GB) or take 2-5 min to load. Check GPU: python -c \"import torch; print('CUDA:', torch.cuda.is_available())\"", flush=True)
    retriever = _create_retriever()

    print(f"Evaluating {len(records)} records...", flush=True)

    for i, rec in enumerate(records):
        fault_codes = rec.get("fault_codes") or []
        desc_parts = []
        if rec.get("repair_summary"):
            desc_parts.append(str(rec["repair_summary"])[:500])
        if rec.get("symptoms"):
            desc_parts.append(str(rec["symptoms"])[:200])
        description = " ".join(desc_parts).strip() or None

        print(f"  [{i+1}/{len(records)}] Retrieving for {fault_codes[:3]}...", flush=True)
        retrieved = _run_retrieval(retriever, fault_codes, description, top_k=10)
        metrics = _compute_metrics(rec["matched_guide_id"], retrieved)

        results.append({
            "scraped_record_id": rec.get("id"),
            "source_url": rec.get("source_url"),
            "fault_codes": fault_codes,
            "description": description,
            "expected_guide_id": rec["matched_guide_id"],
            "expected_guide_title": rec.get("matched_guide_title"),
            "retrieved_guide_ids": metrics["retrieved_ids"],
            "retrieved_scores": [r.get("combined_score", r.get("score", 0)) for r in retrieved[:10]],
            "hit_at_1": metrics["hit_at_1"],
            "hit_at_5": metrics["hit_at_5"],
            "hit_at_10": metrics["hit_at_10"],
            "reciprocal_rank": metrics["reciprocal_rank"],
        })

    n = len(results)
    hit_at_1 = sum(1 for r in results if r["hit_at_1"]) / n if n else 0
    hit_at_5 = sum(1 for r in results if r["hit_at_5"]) / n if n else 0
    hit_at_10 = sum(1 for r in results if r["hit_at_10"]) / n if n else 0
    mrr = sum(r["reciprocal_rank"] for r in results) / n if n else 0

    if persist:
        _persist_to_retrieved_records(db_url, run_id, results)

    print(f"\n=== Retrieval Evaluation (n={n}, run_id={run_id}) ===")
    print(f"  Hit@1:  {hit_at_1:.2%}")
    print(f"  Hit@5:  {hit_at_5:.2%}")
    print(f"  Hit@10: {hit_at_10:.2%}")
    print(f"  MRR:    {mrr:.4f}")

    assert n > 0, "No records evaluated"
    assert hit_at_1 >= 0 and hit_at_10 >= 0, "Invalid metrics"


def test_retrieval_evaluation_cli():
    """Run evaluation via pytest with custom sample size (use -k and env)."""
    # This is a placeholder for pytest collection; actual run uses test_retrieval_evaluation_random_sample
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation using scraped_records. "
        "Uses E5-Mistral-7B — ~1-3 min/record on CPU. Use --sample-size 2 for quick test."
    )
    parser.add_argument("--sample-size", type=int, default=2, help="Number of records (default 2 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--persist", action="store_true", help="Write results to retrieved_records table")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url or not db_url.startswith("postgresql"):
        print("DATABASE_URL (postgresql) required. Set in .env")
        exit(1)

    records = _fetch_eval_records(db_url, sample_size=args.sample_size, seed=args.seed)
    if not records:
        print("No records with matched_guide_id found. Run match_repair_guides.py first.")
        exit(1)

    print("Initializing retriever (E5-Mistral-7B + reranker, may take 2-5 min first run)...", flush=True)
    retriever = _create_retriever()

    results = []
    run_id = str(uuid.uuid4())[:8]
    for i, rec in enumerate(records):
        fault_codes = rec.get("fault_codes") or []
        desc_parts = []
        if rec.get("repair_summary"):
            desc_parts.append(str(rec["repair_summary"])[:500])
        if rec.get("symptoms"):
            desc_parts.append(str(rec["symptoms"])[:200])
        description = " ".join(desc_parts).strip() or None
        print(f"  [{i+1}/{len(records)}] Retrieving for {fault_codes[:3]}...", flush=True)
        retrieved = _run_retrieval(retriever, fault_codes, description, top_k=10)
        metrics = _compute_metrics(rec["matched_guide_id"], retrieved)
        results.append({
            "scraped_record_id": rec.get("id"),
            "source_url": rec.get("source_url"),
            "fault_codes": fault_codes,
            "description": description,
            "expected_guide_id": rec["matched_guide_id"],
            "expected_guide_title": rec.get("matched_guide_title"),
            "retrieved_guide_ids": metrics["retrieved_ids"],
            "retrieved_scores": [r.get("combined_score", r.get("score", 0)) for r in retrieved[:10]],
            "hit_at_1": metrics["hit_at_1"],
            "hit_at_5": metrics["hit_at_5"],
            "hit_at_10": metrics["hit_at_10"],
            "reciprocal_rank": metrics["reciprocal_rank"],
        })

    n = len(results)
    hit_at_1 = sum(1 for r in results if r["hit_at_1"]) / n if n else 0
    hit_at_5 = sum(1 for r in results if r["hit_at_5"]) / n if n else 0
    hit_at_10 = sum(1 for r in results if r["hit_at_10"]) / n if n else 0
    mrr = sum(r["reciprocal_rank"] for r in results) / n if n else 0

    if args.persist:
        _persist_to_retrieved_records(db_url, run_id, results)
        print(f"\nResults persisted to retrieved_records (run_id={run_id})")

    print(f"\n=== Retrieval Evaluation (n={n}, run_id={run_id}) ===")
    print(f"  Hit@1:  {hit_at_1:.2%}")
    print(f"  Hit@5:  {hit_at_5:.2%}")
    print(f"  Hit@10: {hit_at_10:.2%}")
    print(f"  MRR:    {mrr:.4f}")

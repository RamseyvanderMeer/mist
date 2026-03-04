"""
Matcher accuracy evaluation using scraped_records solution column.

Samples records with fault_codes + solution (repair_summary or repair_guide),
runs retrieval, and checks if the top-retrieved guide matches the solution.
- Semantic similarity: embedding-based (always)
- LLM evaluation: OpenAI judges match when OPENAI_API_KEY is set (uses full guide content)

Prints solution and guide title for each record. With --use-llm, pulls full
guide content for LLM evaluation. Uses OPENAI_MODEL from env (default gpt-4o).

Run:
  PYTHONPATH=. python -m pytest tests/test_matcher_accuracy.py -v -s
  OPENAI_API_KEY=xxx python tests/test_matcher_accuracy.py --sample-size 5 --use-llm
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse_fault_codes(fc: Any) -> List[str]:
    if isinstance(fc, str):
        try:
            fc = json.loads(fc) if fc else []
        except json.JSONDecodeError:
            fc = []
    return [str(c).strip() for c in (fc or []) if c and str(c).strip()]


def _solution_text(rec: Dict[str, Any]) -> str:
    """
    Build solution text from repair_guide or repair_summary.
    This is the ground truth: what the forum post says fixed the problem.
    """
    repair_guide = rec.get("repair_guide")
    repair_summary = rec.get("repair_summary") or ""

    if repair_guide:
        if isinstance(repair_guide, str):
            try:
                repair_guide = json.loads(repair_guide) if repair_guide.strip().startswith("{") else repair_guide
            except json.JSONDecodeError:
                pass
        if isinstance(repair_guide, str):
            return repair_guide.strip()
        if isinstance(repair_guide, dict):
            parts = []
            if repair_guide.get("title"):
                parts.append(str(repair_guide["title"]).strip())
            steps = repair_guide.get("procedure_steps") or repair_guide.get("steps")
            if steps:
                if isinstance(steps, list):
                    parts.extend(str(s).strip() for s in steps if s)
                else:
                    parts.append(str(steps).strip())
            return " ".join(parts) if parts else str(repair_guide)

    return repair_summary.strip() if repair_summary else ""


def _fetch_solution_records(
    db_url: str,
    sample_size: int = 20,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch records with fault_codes and solution (repair_summary or repair_guide).
    No matched_guide_id required.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, source_url, fault_codes, repair_summary, repair_guide, symptoms
                FROM scraped_records
                WHERE fault_codes IS NOT NULL AND fault_codes != '[]' AND fault_codes != '{}'
                  AND (
                    (repair_summary IS NOT NULL AND repair_summary != '' AND length(repair_summary) >= 30)
                    OR (repair_guide IS NOT NULL AND repair_guide != '' AND repair_guide != '{}')
                  )
                ORDER BY RANDOM()
                LIMIT 500
            """)
        )
        rows = list(result.fetchall())
        cols = result.keys()

    records = []
    for row in rows:
        rec = dict(zip(cols, row))
        rec["fault_codes"] = _parse_fault_codes(rec.get("fault_codes"))
        if not rec["fault_codes"]:
            continue
        solution = _solution_text(rec)
        if not solution or len(solution) < 20:
            continue
        rec["solution_text"] = solution
        records.append(rec)

    if seed is not None:
        random.seed(seed)
    random.shuffle(records)
    return records[:sample_size]


def _create_retriever():
    from src.retrieval.enhanced_retriever import EnhancedRetriever
    return EnhancedRetriever()


def _create_encoder():
    """Create FaultCodeEncoder for similarity computation (same as retrieval)."""
    import yaml
    from src.paths import get_paths
    from src.embeddings.fault_code_encoder import FaultCodeEncoder

    paths = get_paths()
    with open(paths.embedding_config, "r") as f:
        config = yaml.safe_load(f)
    fc = config.get("models", {}).get("fault_code", {})
    return FaultCodeEncoder(
        model_name=fc.get("model_name", "intfloat/e5-mistral-7b-instruct"),
        device=fc.get("device", "auto"),
        projection_dim=fc.get("projection_dim", 1024),
    )


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a_np = a.cpu().numpy().flatten() if hasattr(a, "cpu") else a.flatten()
    b_np = b.cpu().numpy().flatten() if hasattr(b, "cpu") else b.flatten()
    a_norm = a_np / (np.linalg.norm(a_np) + 1e-9)
    b_norm = b_np / (np.linalg.norm(b_np) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def _is_solution_match(
    encoder,
    solution_text: str,
    retrieved_guide_text: str,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """
    Check if retrieved guide semantically matches the solution.
    Encodes both with same model, computes cosine similarity.
    """
    if not solution_text or not retrieved_guide_text:
        return False, 0.0

    import torch
    with torch.no_grad():
        sol_emb = encoder.encode(solution_text[:2000], normalize=True, is_query=False)
        guide_emb = encoder.encode(retrieved_guide_text[:2000], normalize=True, is_query=False)
    sim = _cosine_similarity(sol_emb, guide_emb)
    return sim >= threshold, sim


def _llm_evaluate_match(
    solution_text: str,
    guide_title: str,
    guide_content: str,
) -> tuple[bool, str]:
    """
    Use OpenAI to evaluate if the repair guide describes the same fix as the solution.
    Requires OPENAI_API_KEY. Uses OPENAI_MODEL from env, else gpt-4o.
    Returns (is_match, reason).
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set"
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)

    prompt = f"""You are an automotive diagnostic expert. A forum post reported a fix for a fault code. We retrieved an ISTA repair guide. Does the guide describe the SAME fix as the forum solution?

Forum solution (what the user did to fix the problem):
---
{solution_text[:1500]}
---

Retrieved repair guide:
Title: {guide_title[:300]}

Content:
---
{(guide_content or "")[:4000]}
---

Answer in this exact format:
MATCH: YES or NO
REASON: One sentence explaining why.
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        response = (resp.choices[0].message.content or "").strip().upper()
        is_match = "MATCH: YES" in response
        reason = ""
        if "REASON:" in response:
            reason = response.split("REASON:")[-1].strip()[:200]
        return is_match, reason
    except Exception as e:
        return False, f"LLM error: {e}"


@pytest.fixture(scope="module")
def db_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "")
    if not url or not url.startswith("postgresql"):
        return None
    return url


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").startswith("postgresql"),
    reason="DATABASE_URL (postgresql) required",
)
def test_matcher_accuracy_by_solution(db_url: Optional[str]):
    """
    Evaluate matcher accuracy: does the retrieved guide match the solution?

    Uses repair_summary/repair_guide as ground truth. Retrieves top guide,
    then checks semantic similarity between solution text and retrieved guide text.
    """
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    sample_size = int(os.environ.get("MATCHER_EVAL_SAMPLE_SIZE", "5"))
    threshold = float(os.environ.get("MATCHER_EVAL_THRESHOLD", "0.55"))
    use_llm = os.environ.get("MATCHER_EVAL_USE_LLM", "").lower() in ("1", "true", "yes")

    records = _fetch_solution_records(db_url, sample_size=sample_size, seed=42)
    if not records:
        pytest.skip("No scraped_records with fault_codes + repair_summary/repair_guide found")

    print(f"Initializing retriever and encoder...", flush=True)
    retriever = _create_retriever()
    encoder = _create_encoder()

    correct = 0
    total = 0
    llm_correct = 0
    similarities: List[float] = []
    use_llm = use_llm and bool(os.environ.get("OPENAI_API_KEY"))
    if use_llm:
        print("LLM evaluation enabled (OPENAI_API_KEY set)", flush=True)

    print(f"Evaluating {len(records)} records (threshold={threshold})...", flush=True)

    for i, rec in enumerate(records):
        fault_codes = rec.get("fault_codes") or []
        solution_text = rec.get("solution_text", "")
        desc_parts = []
        if rec.get("repair_summary"):
            desc_parts.append(str(rec["repair_summary"])[:500])
        if rec.get("symptoms"):
            desc_parts.append(str(rec["symptoms"])[:200])
        description = " ".join(desc_parts).strip() or None

        print(f"  [{i+1}/{len(records)}] Retrieving for {fault_codes[:3]}...", flush=True)

        from src.retrieval.enhanced_retriever import EnhancedRetrieverError
        try:
            retrieved = retriever.retrieve(
                fault_codes=fault_codes,
                obd_data={},
                description=description,
                top_k=3,
            )
        except EnhancedRetrieverError as e:
            pytest.skip(f"Retrieval failed: {e}")

        if not retrieved:
            print(f"    No results", flush=True)
            continue

        top = retrieved[0]
        guide_title = top.get("title") or ""
        guide_content = top.get("text") or ""
        guide_text = guide_title + " " + guide_content[:1000]

        print(f"    Solution: {solution_text[:200]}...", flush=True)
        print(f"    Guide: {guide_title}", flush=True)

        is_match, sim = _is_solution_match(encoder, solution_text, guide_text, threshold=threshold)
        similarities.append(sim)
        if is_match:
            correct += 1
        total += 1

        llm_match = False
        if use_llm:
            llm_match, reason = _llm_evaluate_match(solution_text, guide_title, guide_content)
            if llm_match:
                llm_correct += 1
            print(f"    Similarity={sim:.3f} {'✓' if is_match else '✗'} | LLM: {'✓' if llm_match else '✗'} {reason[:80]}", flush=True)
        else:
            print(f"    Similarity={sim:.3f} {'✓' if is_match else '✗'}", flush=True)

    accuracy = correct / total if total else 0.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    print(f"\n=== Matcher Accuracy (n={total}, threshold={threshold}) ===")
    print(f"  Accuracy (similarity): {accuracy:.2%} ({correct}/{total})")
    print(f"  Avg similarity: {avg_sim:.4f}")
    if use_llm:
        llm_acc = llm_correct / total if total else 0.0
        print(f"  Accuracy (LLM): {llm_acc:.2%} ({llm_correct}/{total})")

    if total == 0:
        pytest.skip(
            "No records evaluated — retrieval returned no results. "
            "Ensure ChromaDB is indexed (run index_repair_guides.py) and encoder loads correctly."
        )
    assert 0 <= accuracy <= 1, "Invalid accuracy"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Matcher accuracy evaluation: compare retrieved guides to solution (repair_summary/repair_guide)"
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55, help="Similarity threshold for match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI to evaluate matches (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url or not db_url.startswith("postgresql"):
        print("DATABASE_URL (postgresql) required. Set in .env")
        exit(1)

    records = _fetch_solution_records(db_url, sample_size=args.sample_size, seed=args.seed)
    if not records:
        print("No records with fault_codes + repair_summary/repair_guide found.")
        exit(1)

    print("Initializing retriever and encoder...", flush=True)
    retriever = _create_retriever()
    encoder = _create_encoder()

    correct = 0
    total = 0
    llm_correct = 0
    similarities: List[float] = []
    use_llm = args.use_llm and bool(os.environ.get("OPENAI_API_KEY"))
    if use_llm:
        print("LLM evaluation enabled (OPENAI_API_KEY set)", flush=True)

    for i, rec in enumerate(records):
        fault_codes = rec.get("fault_codes") or []
        solution_text = rec.get("solution_text", "")
        desc_parts = []
        if rec.get("repair_summary"):
            desc_parts.append(str(rec["repair_summary"])[:500])
        if rec.get("symptoms"):
            desc_parts.append(str(rec["symptoms"])[:200])
        description = " ".join(desc_parts).strip() or None

        print(f"[{i+1}/{len(records)}] Retrieving for {fault_codes[:3]}...", flush=True)
        try:
            retrieved = retriever.retrieve(
                fault_codes=fault_codes,
                obd_data={},
                description=description,
                top_k=3,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if not retrieved:
            continue

        top = retrieved[0]
        guide_title = top.get("title") or ""
        guide_content = top.get("text") or ""
        guide_text = guide_title + " " + guide_content[:1000]

        print(f"  Solution: {solution_text[:200]}...", flush=True)
        print(f"  Guide: {guide_title}", flush=True)

        is_match, sim = _is_solution_match(encoder, solution_text, guide_text, threshold=args.threshold)
        similarities.append(sim)
        if is_match:
            correct += 1
        total += 1

        if use_llm:
            llm_match, reason = _llm_evaluate_match(solution_text, guide_title, guide_content)
            if llm_match:
                llm_correct += 1
            print(f"  Similarity={sim:.3f} {'✓' if is_match else '✗'} | LLM: {'✓' if llm_match else '✗'} {reason[:80]}", flush=True)
        else:
            print(f"  Similarity={sim:.3f} {'✓' if is_match else '✗'}", flush=True)

    accuracy = correct / total if total else 0.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    print(f"\n=== Matcher Accuracy (n={total}) ===")
    print(f"  Accuracy (similarity): {accuracy:.2%} ({correct}/{total})")
    print(f"  Avg similarity: {avg_sim:.4f}")
    if use_llm:
        llm_acc = llm_correct / total if total else 0.0
        print(f"  Accuracy (LLM): {llm_acc:.2%} ({llm_correct}/{total})")

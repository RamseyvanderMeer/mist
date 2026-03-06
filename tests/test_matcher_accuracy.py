"""
Matcher accuracy evaluation using scraped_records solution column.

Samples records with fault_codes + solution (repair_summary or repair_guide), and
checks if the top-retrieved guide matches the solution.
Use MATCHER_EVAL_DB_CANDIDATE_POOL (default 5000) to tune DB prefetch size.
- Semantic similarity: embedding-based (always)
- LLM evaluation: OpenAI judges match when OPENAI_API_KEY is set (uses full guide content)

Default query uses symptom text only (repair_summary + symptoms), with optional
symptom expansion. For an experiment-only baseline, use
MATCHER_EVAL_QUERY_MODE=solution (or --query-mode solution) to include known
solution text in retrieval.

Prints solution and guide title for each record. With --use-llm, pulls full guide
content for LLM evaluation. Uses OPENAI_MODEL from env (default gpt-4o).

Run:
  PYTHONPATH=. python -m pytest tests/test_matcher_accuracy.py -v -s
  OPENAI_API_KEY=xxx python tests/test_matcher_accuracy.py --sample-size 5 --use-llm
  OPENAI_API_KEY=xxx python tests/test_matcher_accuracy.py --compare-modes --sample-size 10 --use-llm
  OPENAI_API_KEY=xxx python tests/test_matcher_accuracy.py --compare-modes --sample-size 10 --progress-interval 2 --use-llm
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

    Uses a capped candidate window instead of ORDER BY RANDOM() for better startup performance.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    try:
        candidate_limit = int(os.environ.get("MATCHER_EVAL_DB_CANDIDATE_POOL", "5000"))
    except ValueError:
        candidate_limit = 5000
    if candidate_limit <= 0:
        candidate_limit = 5000
    if candidate_limit < sample_size:
        candidate_limit = sample_size * 4
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


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _create_query_expander(enabled: bool = True):
    """Create optional symptom expander. Returns None if unavailable."""
    if not enabled:
        return None
    try:
        from src.retrieval.query_expander import QueryExpander

        return QueryExpander()
    except Exception:
        return None


def _build_query_description(
    rec: Dict[str, Any],
    query_mode: str = "symptom",
    query_expander=None,
) -> Optional[str]:
    """
    Build retrieval query text from record data.

    - symptom mode: use repair_summary + symptoms then optional symptom expansion
    - solution mode: include ground-truth solution text (benchmark-only)
    """
    solution_text = rec.get("solution_text", "") or ""
    parts: List[str] = []

    if rec.get("repair_summary"):
        parts.append(str(rec["repair_summary"])[:500])
    if rec.get("symptoms"):
        parts.append(str(rec["symptoms"])[:200])
    if query_mode == "solution" and solution_text:
        parts.append(f"Fix: {str(solution_text)[:300]}")

    description = " ".join(parts).strip()
    if not description:
        return None

    if query_mode == "symptom" and query_expander is not None:
        try:
            expanded = query_expander.expand_symptom_for_search(description)
            return (expanded or description).strip() or None
        except Exception:
            return description

    return description


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


def _run_matcher_evaluation(
    records: List[Dict[str, Any]],
    retriever,
    encoder,
    query_mode: str = "symptom",
    threshold: float = 0.55,
    use_llm: bool = False,
    use_symptom_expansion: bool = True,
    top_k: int = 3,
    verbose: bool = True,
    mode_label: str = "",
    progress_interval: int = 1,
) -> Dict[str, Any]:
    """Run matcher accuracy evaluation for one query mode."""
    from src.retrieval.enhanced_retriever import EnhancedRetrieverError

    query_mode = query_mode.strip().lower()
    if query_mode not in {"symptom", "solution"}:
        query_mode = "symptom"
    if not mode_label:
        mode_label = query_mode

    query_expander = _create_query_expander(
        enabled=bool(query_mode == "symptom" and use_symptom_expansion)
    )
    if verbose:
        if query_mode == "solution":
            print("Query mode: solution benchmark (includes solution text)", flush=True)
        else:
            print(
                f"Query mode [{mode_label}]: symptom-only"
                f" (symptom expansion {'on' if query_expander is not None else 'off'})",
                flush=True,
            )

    correct = 0
    total = 0
    llm_correct = 0
    similarities: List[float] = []
    no_results = 0
    retrieval_errors = 0

    total_records = len(records)
    try:
        progress_interval = int(progress_interval)
    except Exception:
        progress_interval = 1

    if progress_interval <= 0:
        progress_interval = total_records + 1

    show_detailed_records = verbose and progress_interval == 1

    for i, rec in enumerate(records):
        fault_codes = rec.get("fault_codes") or []
        solution_text = rec.get("solution_text", "")
        description = _build_query_description(
            rec=rec,
            query_mode=query_mode,
            query_expander=query_expander,
        )

        should_print_progress = verbose and (
            show_detailed_records
            or (i + 1) % progress_interval == 0
            or (i + 1) == total_records
        )

        if should_print_progress:
            print(f"  [{i+1}/{len(records)}] Retrieving for {fault_codes[:3]}...", flush=True)

        try:
            retrieved = retriever.retrieve(
                fault_codes=fault_codes,
                obd_data={},
                description=description,
                top_k=top_k,
            )
        except EnhancedRetrieverError as e:
            retrieval_errors += 1
            if verbose:
                print(f"    Retrieval error: {e}")
            continue

        if not retrieved:
            no_results += 1
            if should_print_progress:
                print("    No results")
            continue

        top = retrieved[0]
        guide_title = top.get("title") or ""
        guide_content = top.get("text") or ""
        guide_text = guide_title + " " + guide_content[:1000]

        if show_detailed_records:
            print(f"    Solution: {solution_text[:200]}...", flush=True)
            print(f"    Guide: {guide_title}", flush=True)

        is_match, sim = _is_solution_match(
            encoder, solution_text, guide_text, threshold=threshold
        )
        similarities.append(sim)
        if is_match:
            correct += 1
        total += 1

        llm_match = False
        reason = ""
        if use_llm:
            llm_match, reason = _llm_evaluate_match(solution_text, guide_title, guide_content)
            if llm_match:
                llm_correct += 1

        if show_detailed_records:
            if use_llm:
                print(
                    f"    Similarity={sim:.3f} {'✓' if is_match else '✗'} |"
                    f" LLM: {'✓' if llm_match else '✗'} {reason[:80]}",
                    flush=True,
                )
            else:
                print(f"    Similarity={sim:.3f} {'✓' if is_match else '✗'}", flush=True)

    attempted = len(records)
    accuracy = correct / total if total else 0.0
    accuracy_overall = correct / attempted if attempted else 0.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    llm_accuracy = llm_correct / total if use_llm and total else 0.0
    llm_accuracy_overall = llm_correct / attempted if use_llm and attempted else 0.0
    coverage = total / attempted if attempted else 0.0

    return {
        "mode": query_mode,
        "label": mode_label or query_mode,
        "top_k": top_k,
        "threshold": threshold,
        "attempted": attempted,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_overall": accuracy_overall,
        "avg_similarity": avg_sim,
        "use_llm": use_llm,
        "llm_correct": llm_correct,
        "llm_accuracy": llm_accuracy,
        "llm_accuracy_overall": llm_accuracy_overall,
        "coverage": coverage,
        "no_results": no_results,
        "retrieval_errors": retrieval_errors,
    }


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
def test_matcher_accuracy_by_solution(db_url: Optional[str], pytestconfig):
    """
    Evaluate matcher accuracy: does the retrieved guide match the solution?

    Uses repair_summary/repair_guide as ground truth. Retrieves top guide,
    then checks semantic similarity between solution text and retrieved guide text.
    """
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    sample_size = int(os.environ.get("MATCHER_EVAL_SAMPLE_SIZE", "5"))
    threshold = float(os.environ.get("MATCHER_EVAL_THRESHOLD", "0.55"))
    use_llm = (
        pytestconfig.getoption("use_llm")
        or os.environ.get("MATCHER_EVAL_USE_LLM", "").lower() in ("1", "true", "yes")
    )
    query_mode = os.environ.get("MATCHER_EVAL_QUERY_MODE", "symptom").strip().lower()
    if query_mode not in {"symptom", "solution"}:
        query_mode = "symptom"
    use_symptom_expansion = query_mode == "symptom" and _to_bool(
        os.environ.get("MATCHER_EVAL_USE_SYMPTOM_EXPANSION"), True
    )
    use_llm = bool(use_llm) and bool(os.environ.get("OPENAI_API_KEY"))

    records = _fetch_solution_records(db_url, sample_size=sample_size, seed=42)
    if not records:
        pytest.skip("No scraped_records with fault_codes + repair_summary/repair_guide found")

    print(f"Initializing retriever and encoder...", flush=True)
    retriever = _create_retriever()
    encoder = _create_encoder()
    if use_llm:
        print("LLM evaluation enabled (OPENAI_API_KEY set)", flush=True)

    print(f"Evaluating {len(records)} records (threshold={threshold})...", flush=True)

    result = _run_matcher_evaluation(
        records=records,
        retriever=retriever,
        encoder=encoder,
        query_mode=query_mode,
        threshold=threshold,
        use_llm=use_llm,
        use_symptom_expansion=use_symptom_expansion,
        top_k=3,
        verbose=True,
        mode_label="cli",
    )
    total = result["total"]
    attempted = result.get("attempted", total)
    accuracy = result["accuracy"]
    accuracy_overall = result.get("accuracy_overall", accuracy)
    avg_sim = result["avg_similarity"]
    llm_correct = result["llm_correct"]

    print(f"\n=== Matcher Accuracy (attempted={attempted}, evaluated={total}, threshold={threshold}) ===")
    print(f"  Mode: {result['mode']}")
    print(f"  Accuracy (similarity, evaluated): {accuracy:.2%} ({result['correct']}/{total})")
    print(f"  Accuracy (similarity, overall):   {accuracy_overall:.2%} ({result['correct']}/{attempted})")
    print(f"  Coverage (evaluated/attempted):   {result.get('coverage', 0.0):.2%}")
    print(f"  Avg similarity: {avg_sim:.4f}")
    if use_llm:
        llm_acc = result["llm_accuracy"]
        llm_acc_overall = result.get("llm_accuracy_overall", llm_acc)
        print(f"  Accuracy (LLM, evaluated): {llm_acc:.2%} ({llm_correct}/{total})")
        print(f"  Accuracy (LLM, overall):   {llm_acc_overall:.2%} ({llm_correct}/{attempted})")

    if total == 0:
        pytest.skip(
            "No records evaluated — retrieval returned no results. "
            "Ensure ChromaDB is indexed (run index_repair_guides.py) and encoder loads correctly."
        )
    assert 0 <= accuracy <= 1, "Invalid accuracy"


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").startswith("postgresql"),
    reason="DATABASE_URL (postgresql) required",
)
def test_matcher_accuracy_query_mode_delta(db_url: Optional[str], pytestconfig):
    """
    Compare symptom-only vs solution-benchmark query modes and report the delta.

    This is intended as a regression-style check that tracks semantic-to-lexical
    improvements after retrieval changes.
    """
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    sample_size = int(os.environ.get("MATCHER_EVAL_SAMPLE_SIZE", "5"))
    threshold = float(os.environ.get("MATCHER_EVAL_THRESHOLD", "0.55"))
    use_symptom_expansion = _to_bool(os.environ.get("MATCHER_EVAL_USE_SYMPTOM_EXPANSION"), True)
    use_llm = (
        pytestconfig.getoption("use_llm")
        or os.environ.get("MATCHER_EVAL_USE_LLM", "").lower() in ("1", "true", "yes")
    )
    use_llm = use_llm and bool(os.environ.get("OPENAI_API_KEY"))

    sample_seed = int(os.environ.get("MATCHER_EVAL_SEED", "42"))
    records = _fetch_solution_records(db_url, sample_size=sample_size, seed=sample_seed)
    if not records:
        pytest.skip("No scraped_records with fault_codes + repair_summary/repair_guide found")

    print("Initializing retriever and encoder for delta comparison...", flush=True)
    retriever = _create_retriever()
    encoder = _create_encoder()

    print(f"Running symptom-only mode on {len(records)} records...", flush=True)
    symptom_result = _run_matcher_evaluation(
        records=records,
        retriever=retriever,
        encoder=encoder,
        query_mode="symptom",
        threshold=threshold,
        use_llm=use_llm,
        use_symptom_expansion=use_symptom_expansion,
        top_k=3,
        verbose=False,
        mode_label="symptom",
    )

    print(f"Running solution-benchmark mode on {len(records)} records...", flush=True)
    solution_result = _run_matcher_evaluation(
        records=records,
        retriever=retriever,
        encoder=encoder,
        query_mode="solution",
        threshold=threshold,
        use_llm=use_llm,
        use_symptom_expansion=False,
        top_k=3,
        verbose=False,
        mode_label="solution",
    )

    if symptom_result["total"] == 0 and solution_result["total"] == 0:
        pytest.skip(
            "No records evaluated in either mode — retrieval/index issues likely."
            " Ensure ChromaDB is indexed and encoder loads correctly."
        )

    delta_similarity = solution_result["accuracy_overall"] - symptom_result["accuracy_overall"]
    print("\n=== Matcher Accuracy Delta (symptom vs solution) ===")
    print(
        f"  Symptom mode (overall): {symptom_result['accuracy_overall']:.2%}"
        f" ({symptom_result['correct']}/{symptom_result['attempted']})"
    )
    print(
        f"  Solution mode (overall): {solution_result['accuracy_overall']:.2%}"
        f" ({solution_result['correct']}/{solution_result['attempted']})"
    )
    print(
        f"  Similarity accuracy delta (solution - symptom): {delta_similarity:+.2%}"
    )
    print(
        f"  Coverage symptom: {symptom_result.get('coverage', 0.0):.2%} | "
        f"solution: {solution_result.get('coverage', 0.0):.2%}"
    )

    if use_llm:
        delta_llm = solution_result["llm_accuracy_overall"] - symptom_result["llm_accuracy_overall"]
        print(f"  LLM accuracy (symptom, overall): {symptom_result['llm_accuracy_overall']:.2%}")
        print(f"  LLM accuracy (solution, overall): {solution_result['llm_accuracy_overall']:.2%}")
        print(f"  LLM accuracy delta (solution - symptom): {delta_llm:+.2%}")

    assert 0 <= symptom_result["accuracy"] <= 1
    assert 0 <= solution_result["accuracy"] <= 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Matcher accuracy evaluation: compare retrieved guides to solution (repair_summary/repair_guide)"
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55, help="Similarity threshold for match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query-mode",
        choices=("symptom", "solution", "both"),
        default="symptom",
        help="symptom (default): repair_summary/symptoms; solution: benchmark with solution text; both: A/B comparison"
    )
    parser.add_argument(
        "--disable-symptom-expansion",
        action="store_true",
        help="Skip symptom expansion in symptom mode"
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run symptom + solution modes and print delta comparison"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print progress every N records. Set 1 for every record, 0 for only final record.",
    )
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
    query_mode = args.query_mode
    use_symptom_expansion = (
        (query_mode == "symptom" or query_mode == "both")
        and not args.disable_symptom_expansion
        and _to_bool(os.environ.get("MATCHER_EVAL_USE_SYMPTOM_EXPANSION"), True)
    )
    use_llm = args.use_llm and bool(os.environ.get("OPENAI_API_KEY"))
    if args.use_llm and not bool(os.environ.get("OPENAI_API_KEY")):
        print(
            "LLM mode requested (--use-llm) but OPENAI_API_KEY is not set; "
            "falling back to similarity-only metrics.",
            flush=True,
        )
    run_both = args.compare_modes or query_mode == "both"
    if use_llm:
        print("LLM evaluation enabled (OPENAI_API_KEY set)", flush=True)

    if run_both:
        print("Running A/B comparison (symptom vs solution)...", flush=True)

        symptom_result = _run_matcher_evaluation(
            records=records,
            retriever=retriever,
            encoder=encoder,
            query_mode="symptom",
            threshold=args.threshold,
            use_llm=use_llm,
            use_symptom_expansion=use_symptom_expansion,
            top_k=3,
            verbose=True,
            progress_interval=args.progress_interval,
            mode_label="symptom",
        )

        solution_result = _run_matcher_evaluation(
            records=records,
            retriever=retriever,
            encoder=encoder,
            query_mode="solution",
            threshold=args.threshold,
            use_llm=use_llm,
            use_symptom_expansion=False,
            top_k=3,
            verbose=True,
            progress_interval=args.progress_interval,
            mode_label="solution",
        )

        if symptom_result["total"] == 0 and solution_result["total"] == 0:
            print(
                "No records evaluated in either mode — retrieval/index issues likely. "
                "Ensure ChromaDB is indexed (run index_repair_guides.py) and encoder loads correctly.",
                flush=True,
            )
            exit(1)

        delta_similarity = solution_result["accuracy_overall"] - symptom_result["accuracy_overall"]
        print("\n=== Matcher A/B Delta (script) ===")
        print(
            f"  Symptom mode (overall): {symptom_result['accuracy_overall']:.2%} "
            f"({symptom_result['correct']}/{symptom_result['attempted']})"
        )
        print(
            f"  Solution mode (overall): {solution_result['accuracy_overall']:.2%} "
            f"({solution_result['correct']}/{solution_result['attempted']})"
        )
        print(
            f"  Similarity accuracy delta (solution - symptom): {delta_similarity:+.2%}"
        )
        print(
            f"  Coverage symptom: {symptom_result.get('coverage', 0.0):.2%} | "
            f"solution: {solution_result.get('coverage', 0.0):.2%}"
        )

        if use_llm:
            delta_llm = solution_result["llm_accuracy_overall"] - symptom_result["llm_accuracy_overall"]
            print(f"  LLM accuracy (symptom, overall): {symptom_result['llm_accuracy_overall']:.2%}")
            print(f"  LLM accuracy (solution, overall): {solution_result['llm_accuracy_overall']:.2%}")
            print(f"  LLM accuracy delta (solution - symptom): {delta_llm:+.2%}")
    else:
        print(f"Running single mode: {query_mode}", flush=True)
        result = _run_matcher_evaluation(
            records=records,
            retriever=retriever,
            encoder=encoder,
            query_mode=query_mode,
            threshold=args.threshold,
            use_llm=use_llm,
            use_symptom_expansion=use_symptom_expansion,
            top_k=3,
            verbose=True,
            progress_interval=args.progress_interval,
            mode_label="script-single",
        )

        total = result["total"]
        attempted = result.get("attempted", total)
        accuracy = result["accuracy"]
        accuracy_overall = result.get("accuracy_overall", accuracy)
        avg_sim = result["avg_similarity"]
        print(f"\n=== Matcher Accuracy (attempted={attempted}, evaluated={total}) ===")
        print(f"  Mode: {query_mode}")
        print(f"  Accuracy (similarity, evaluated): {accuracy:.2%} ({result['correct']}/{total})")
        print(f"  Accuracy (similarity, overall):   {accuracy_overall:.2%} ({result['correct']}/{attempted})")
        print(f"  Coverage (evaluated/attempted):   {result.get('coverage', 0.0):.2%}")
        print(f"  Avg similarity: {avg_sim:.4f}")
        if use_llm:
            llm_acc = result["llm_accuracy"]
            llm_acc_overall = result.get("llm_accuracy_overall", llm_acc)
            print(f"  Accuracy (LLM, evaluated): {llm_acc:.2%} ({result['llm_correct']}/{total})")
            print(f"  Accuracy (LLM, overall):   {llm_acc_overall:.2%} ({result['llm_correct']}/{attempted})")

        if total == 0:
            print(
                "No records evaluated — retrieval returned no results. "
                "Ensure ChromaDB is indexed (run index_repair_guides.py) and encoder loads correctly.",
                flush=True,
            )
            exit(1)

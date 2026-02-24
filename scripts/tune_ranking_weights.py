#!/usr/bin/env python3
"""
Tune ranking weights using scraped_records evaluation data.

Loads evaluation pairs from scraped_records (fault_codes + repair_summary vs matched_guide_id),
runs grid search over embedding_similarity, rerank_score, kg_path_score, feedback_score,
and INSERT/UPDATEs the best weights into ranking_weights table.

Usage:
    python scripts/tune_ranking_weights.py
    python scripts/tune_ranking_weights.py --name tuned_v1
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_evaluation_pairs(db_url: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Load (fault_codes, repair_summary, matched_guide_id) from scraped_records for evaluation."""
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    pairs = []
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT fault_codes, repair_summary, matched_guide_id
                FROM scraped_records
                WHERE outcome IN ('success', 'partial')
                  AND repair_summary IS NOT NULL
                  AND matched_guide_id IS NOT NULL
                LIMIT :limit
            """),
            {"limit": limit}
        )
        for row in result:
            fc = row[0]
            if isinstance(fc, str):
                try:
                    fc = json.loads(fc) if fc else []
                except json.JSONDecodeError:
                    fc = []
            pairs.append({
                "fault_codes": fc,
                "repair_summary": row[1],
                "matched_guide_id": row[2],
            })
    return pairs


def evaluate_weights(
    weights: Dict[str, float],
    pairs: List[Dict[str, Any]],
    vector_store: Any,
    encoder: Any,
) -> float:
    """
    Simple evaluation: for each pair, check if matched_guide_id is in top-k retrieval.
    Returns fraction of pairs where match is in top-5.
    """
    correct = 0
    for pair in pairs:
        try:
            summary = (pair.get("repair_summary") or "")[:1000]
            if not summary:
                continue
            emb = encoder.encode(summary, normalize=True)
            if emb.dim() > 1:
                emb = emb.squeeze(0)
            emb_np = emb.detach().cpu().numpy()
            results = vector_store.search(query_embedding=emb_np, top_k=5)
            ids = [r.get("procedure_id") for r in results if r.get("procedure_id")]
            if pair.get("matched_guide_id") in ids:
                correct += 1
        except Exception as e:
            logger.debug("Eval error: %s", e)
    return correct / len(pairs) if pairs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Tune ranking weights from scraped_records")
    parser.add_argument("--name", type=str, default="tuned", help="Name for weights row")
    parser.add_argument("--limit", type=int, default=500, help="Max evaluation pairs")
    parser.add_argument("--grid-size", type=int, default=3, help="Grid points per weight (2 or 3)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url or not db_url.startswith("postgresql"):
        logger.error("DATABASE_URL required. Set in .env")
        sys.exit(1)

    pairs = get_evaluation_pairs(db_url, limit=args.limit)
    if len(pairs) < 10:
        logger.warning("Few evaluation pairs (%d). Need more matched records.", len(pairs))
        sys.exit(1)
    logger.info("Loaded %d evaluation pairs", len(pairs))

    try:
        import yaml
        from embeddings.fault_code_encoder import FaultCodeEncoder
        from retrieval.vector_store import VectorStore
        from paths import get_paths
        paths = get_paths()
        with open(paths.retrieval_config, "r") as f:
            config = yaml.safe_load(f)
        encoder = FaultCodeEncoder()
        vector_store = VectorStore(config.get("vector_store", {}))
    except Exception as e:
        logger.error("Failed to init encoder/vector_store: %s", e)
        sys.exit(1)

    best_score = 0.0
    best_weights = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1,
    }
    step = 0.1 if args.grid_size == 2 else 0.15
    for es in [0.3, 0.4, 0.5]:
        for rs in [0.2, 0.3, 0.4]:
            for kg in [0.1, 0.2, 0.3]:
                fb = 1.0 - es - rs - kg
                if fb < 0 or fb > 0.5:
                    continue
                w = {
                    "embedding_similarity": es,
                    "rerank_score": rs,
                    "kg_path_score": kg,
                    "feedback_score": fb,
                }
                score = evaluate_weights(w, pairs, vector_store, encoder)
                if score > best_score:
                    best_score = score
                    best_weights = w
                    logger.info("New best: %.3f with %s", score, w)

    logger.info("Best score: %.3f, weights: %s", best_score, best_weights)

    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ranking_weights (name, embedding_similarity, rerank_score, kg_path_score, feedback_score, is_active)
                VALUES (:name, :es, :rs, :kg, :fb, true)
                ON CONFLICT (name) DO UPDATE SET
                    embedding_similarity = EXCLUDED.embedding_similarity,
                    rerank_score = EXCLUDED.rerank_score,
                    kg_path_score = EXCLUDED.kg_path_score,
                    feedback_score = EXCLUDED.feedback_score
            """),
            {
                "name": args.name,
                "es": best_weights["embedding_similarity"],
                "rs": best_weights["rerank_score"],
                "kg": best_weights["kg_path_score"],
                "fb": best_weights["feedback_score"],
            }
        )
        conn.commit()
    logger.info("Saved weights as '%s' in ranking_weights", args.name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug script: run each step of retrieval evaluation to find where it hangs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def step1_check_env():
    print("Step 1: Check environment...", flush=True)
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url or not db_url.startswith("postgresql"):
        print("  FAIL: DATABASE_URL not set or not postgresql")
        return None
    print("  OK: DATABASE_URL set")
    return db_url


def step2_fetch_records(db_url):
    print("Step 2a: Connect to Postgres...", flush=True)
    from sqlalchemy import create_engine, text
    # Add connect_timeout for Neon (serverless can cold-start)
    engine = create_engine(
        db_url,
        connect_args={"connect_timeout": 30} if "postgresql" in db_url else {},
        pool_pre_ping=True,
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("  OK: Connection works")

    print("Step 2b: Fetch records from scraped_records...", flush=True)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, source_url, fault_codes, repair_summary, symptoms,
                   matched_guide_id, matched_guide_title
            FROM scraped_records
            WHERE matched_guide_id IS NOT NULL AND matched_guide_id != ''
              AND (repair_summary IS NOT NULL AND repair_summary != ''
                   OR symptoms IS NOT NULL AND symptoms != '')
              AND fault_codes IS NOT NULL AND fault_codes != '[]'
            LIMIT 2
        """))
        rows = list(result.fetchall())
    print(f"  OK: Fetched {len(rows)} records")
    return rows


def step3_chromadb():
    print("Step 3: Connect to ChromaDB...", flush=True)
    from src.retrieval.vector_store import VectorStore
    import yaml
    from src.paths import get_paths
    paths = get_paths()
    with open(paths.retrieval_config, "r") as f:
        config = yaml.safe_load(f)
    store = VectorStore(config["vector_store"])
    info = store.get_collection_info()
    print(f"  OK: ChromaDB connected, {info.get('vectors_count', 0)} vectors")
    return store


def step4_encoder():
    print("Step 4: Load FaultCodeEncoder (E5-Mistral-7B)...", flush=True)
    import torch
    print(f"  torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    from src.embeddings.fault_code_encoder import FaultCodeEncoder
    import yaml
    from src.paths import get_paths
    paths = get_paths()
    with open(paths.embedding_config, "r") as f:
        config = yaml.safe_load(f)
    fc = config.get("models", {}).get("fault_code", {})
    encoder = FaultCodeEncoder(
        model_name=fc.get("model_name", "intfloat/e5-mistral-7b-instruct"),
        device=fc.get("device", "auto"),
        projection_dim=fc.get("projection_dim", 1024),
    )
    print(f"  OK: Encoder loaded on {encoder.device}")
    return encoder


def step5_reranker():
    print("Step 5: Load Reranker (cross-encoder)...", flush=True)
    from src.retrieval.reranker import Reranker
    r = Reranker()
    print(f"  OK: Reranker loaded, enabled={r.enabled}")
    return r


def step6_kg():
    print("Step 6: Load Knowledge Graph...", flush=True)
    from src.knowledge.graph_query import KnowledgeGraphQuery
    from src.paths import get_paths
    paths = get_paths()
    kg_path = paths.knowledge_graph
    kg = KnowledgeGraphQuery(kg_path)
    print(f"  OK: KG loaded, {kg.graph.number_of_nodes()} nodes")
    return kg


def step7_full_retriever():
    print("Step 7: Create EnhancedRetriever (all components)...", flush=True)
    from src.retrieval.enhanced_retriever import EnhancedRetriever
    retriever = EnhancedRetriever()
    print("  OK: EnhancedRetriever created")
    return retriever


def step8_retrieve(retriever, fault_codes, description):
    print("Step 8: Run single retrieval...", flush=True)
    results = retriever.retrieve(
        fault_codes=fault_codes,
        obd_data={},
        description=description,
        top_k=5,
    )
    print(f"  OK: Retrieved {len(results)} results")
    return results


def main():
    print("=" * 60, flush=True)
    print("Retrieval Evaluation Debug - running each step", flush=True)
    print("=" * 60, flush=True)

    db_url = step1_check_env()
    if not db_url:
        sys.exit(1)

    rows = step2_fetch_records(db_url)
    if not rows:
        print("  No records with matched_guide_id. Run match_repair_guides.py first.")
        sys.exit(1)

    print("", flush=True)
    step3_chromadb()
    print("", flush=True)
    step4_encoder()
    step5_reranker()
    step6_kg()
    retriever = step7_full_retriever()

    # Parse first record and run retrieval
    rec = dict(zip(
        ["id", "source_url", "fault_codes", "repair_summary", "symptoms", "matched_guide_id", "matched_guide_title"],
        rows[0]
    ))
    import json
    fc = rec.get("fault_codes")
    if isinstance(fc, str):
        fc = json.loads(fc) if fc else []
    fault_codes = [str(c).strip() for c in (fc or []) if c]
    desc = (rec.get("repair_summary") or "")[:200] or (rec.get("symptoms") or "")[:200] or None

    step8_retrieve(retriever, fault_codes, desc)

    print("=" * 60)
    print("All steps completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

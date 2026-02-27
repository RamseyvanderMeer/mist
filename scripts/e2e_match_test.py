#!/usr/bin/env python3
"""
End-to-end test: add a known procedure to ChromaDB, search with a repair summary, verify match.

This script:
1. Creates a temporary test collection
2. Encodes one document with is_query=False (document encoding)
3. Searches with a semantically similar repair summary (is_query=True)
4. Verifies we get a high similarity score (>0.6)

Run after projection fix. If this passes, re-index and run match_repair_guides.
"""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Test pair: repair summary (query) -> procedure text (document)
# These should be semantically similar for a strong match
TEST_QUERY = "Replaced ignition coil on cylinder 3. Misfire resolved."
TEST_DOCUMENT = """Replace ignition coil

Procedure: Replace ignition coil on cylinder 3.
Remove ignition coil connector. Remove ignition coil. Install new ignition coil.
Tighten to specification. Reconnect connector. Clear fault codes."""

TEST_COLLECTION = "repair_guides_e2e_test"


def main():
    print("=" * 60)
    print("E2E Match Test: Add document -> Search -> Verify score")
    print("=" * 60)

    # Load config
    from paths import get_paths
    paths = get_paths()
    with open(paths.retrieval_config, "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)

    vs_config = retrieval_config.get("vector_store", {})
    vs_config = dict(vs_config)
    vs_config["collection_name"] = TEST_COLLECTION

    print("\n1. Loading FaultCodeEncoder...")
    from embeddings.fault_code_encoder import FaultCodeEncoder

    embedding_config_path = ROOT / "config" / "embedding_config.yaml"
    with open(embedding_config_path, "r", encoding="utf-8") as f:
        emb_config = yaml.safe_load(f)
    fc_config = emb_config.get("models", {}).get("fault_code", {})
    encoder = FaultCodeEncoder(
        model_name=fc_config.get("model_name", "intfloat/e5-mistral-7b-instruct"),
        device=fc_config.get("device", "cpu"),
        projection_dim=fc_config.get("projection_dim", 768),
    )

    print("\n2. Initializing VectorStore (test collection)...")
    from retrieval.vector_store import VectorStore

    store = VectorStore(vs_config)

    print("\n3. Encoding document (is_query=False)...")
    import torch

    with torch.no_grad():
        doc_emb = encoder.encode(TEST_DOCUMENT, normalize=True, is_query=False)
    doc_emb_np = doc_emb.cpu().numpy()
    if doc_emb_np.ndim > 1:
        doc_emb_np = doc_emb_np.squeeze(0)

    print("\n4. Adding document to ChromaDB...")
    test_id = uuid.uuid4()
    doc = {
        "id": test_id,
        "text": TEST_DOCUMENT,
        "title": "Replace ignition coil",
        "procedure_id": "e2e-test-001",
        "procedure_name": "Replace ignition coil",
        "fault_codes": ["P0303"],
        "ecu_category": "",
        "metadata": {"e2e_test": True},
    }
    store.add(np.expand_dims(doc_emb_np, 0), [doc], batch_size=1)

    print("\n5. Encoding query (is_query=True)...")
    with torch.no_grad():
        query_emb = encoder.encode(TEST_QUERY, normalize=True, is_query=True)
    query_emb_np = query_emb.cpu().numpy()
    if query_emb_np.ndim > 1:
        query_emb_np = query_emb_np.squeeze(0)

    print("\n6. Searching...")
    results = store.search(
        query_embedding=query_emb_np,
        top_k=5,
        filter_dict=None,
    )

    print("\n7. Results:")
    if not results:
        print("  FAIL: No results returned!")
        _cleanup(store)
        return 1

    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.4f}  title={r.get('title', '')[:50]}")

    best_score = results[0]["score"]
    threshold = 0.6
    if best_score >= threshold:
        print(f"\n  PASS: Best score {best_score:.4f} >= {threshold}")
        print("  Re-index main collection: python scripts/index_repair_guides.py --no-resume")
        print("  Then run: python scripts/match_repair_guides.py")
    else:
        print(f"\n  FAIL: Best score {best_score:.4f} < {threshold}")
        print("  Check: projection seed, query vs document encoding, re-index.")

    _cleanup(store)
    return 0 if best_score >= threshold else 1


def _cleanup(store):
    """Delete test collection."""
    try:
        store.client.delete_collection(TEST_COLLECTION)
        print(f"\n  Cleaned up test collection: {TEST_COLLECTION}")
    except Exception as e:
        print(f"\n  Warning: Could not delete test collection: {e}")


if __name__ == "__main__":
    sys.exit(main())

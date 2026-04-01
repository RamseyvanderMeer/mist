"""
Quick test script for repair guide matching using HuggingFace Inference API.
Tests vector search on a small subset of guides without needing local GPU.
"""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings.hf_encoder import HuggingFaceEncoder
from src.retrieval.vector_store import VectorStore
from src.database.ista_db import IstaDatabase
from src.paths import get_paths


def test_guide_matching(
    query: str = "engine misfire cylinder 1",
    top_k: int = 5,
    max_guides: int = 100
):
    """
    Test repair guide matching with HF API embeddings.
    
    Args:
        query: Search query
        top_k: Number of results to return
        max_guides: Max guides to index for testing
    """
    print(f"Testing guide matching for query: '{query}'")
    print(f"Indexing up to {max_guides} guides for testing...")
    
    # Initialize encoder
    encoder = HuggingFaceEncoder(
        model_name="intfloat/e5-mistral-7b-instruct",
        projection_dim=768,
        use_api=True
    )
    
    # Load ISTA database
    paths = get_paths()
    db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
    
    if not db_path.exists():
        print(f"ISTA database not found at {db_path}")
        print("Using sample data instead...")
        guides = [
            {"id": "TEST001", "title": "Engine Misfire Diagnosis", "content": "Check spark plugs, ignition coils, and fuel injectors for cylinder 1 misfire."},
            {"id": "TEST002", "title": "Oxygen Sensor Replacement", "content": "Replace oxygen sensor bank 1 sensor 2 when slow response detected."},
            {"id": "TEST003", "title": "Catalytic Converter Efficiency", "content": "P0420 code indicates catalytic converter efficiency below threshold."},
        ]
    else:
        db = IstaDatabase(db_path=str(db_path))
        guides = db.get_repair_procedures(limit=max_guides)
        db.close()
        print(f"Loaded {len(guides)} guides from ISTA database")
    
    # Index guides
    print("Encoding guides...")
    texts = [f"{g.get('title', '')}\n{g.get('content', g.get('text', ''))}" for g in guides]
    embeddings = encoder.encode(texts, normalize=True, is_query=False)
    
    print(f"Encoded {len(guides)} guides to {embeddings.shape}")
    
    # Encode query
    print("Encoding query...")
    query_embedding = encoder.encode([query], normalize=True, is_query=True)
    
    # Simple cosine similarity search
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nTop {top_k} matching guides:")
    print("=" * 60)
    for i, idx in enumerate(top_indices, 1):
        guide = guides[idx]
        score = similarities[idx]
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   ID: {guide.get('id', guide.get('procedure_id', 'N/A'))}")
        print(f"   Title: {guide.get('title', 'N/A')}")
        print(f"   Content preview: {guide.get('content', guide.get('text', 'N/A'))[:150]}...")
    
    return guides, embeddings


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Test repair guide matching with HF API")
    ap.add_argument("--query", default="engine misfire cylinder 1", help="Search query")
    ap.add_argument("--top-k", type=int, default=5, help="Number of results")
    ap.add_argument("--max-guides", type=int, default=100, help="Max guides to index")
    args = ap.parse_args()
    
    # Check for HF token
    if not os.environ.get("HUGGINGFACE_API_TOKEN"):
        print("WARNING: HUGGINGFACE_API_TOKEN not set!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Then run: export HUGGINGFACE_API_TOKEN='your-token-here'")
        print("\nFalling back to local model (all-MiniLM-L6-v2)...")
    
    test_guide_matching(
        query=args.query,
        top_k=args.top_k,
        max_guides=args.max_guides
    )

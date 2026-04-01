"""
Test repair guide matching with existing vector store.
Uses the 400 already-indexed guides for testing.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pickle

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.vector_store import VectorStore


def test_guide_matching(query: str = "engine misfire cylinder 1", top_k: int = 5):
    """Test matching against existing vector store."""
    print(f"Testing guide matching for query: '{query}'")
    print("=" * 60)
    
    # Load vector store
    config = {
        "type": "chroma",
        "collection_name": "repair_guides_enhanced",
        "persist_directory": str(ROOT / "data" / "vector_store")
    }
    
    print("Loading vector store...")
    store = VectorStore(config)
    
    # Get collection info
    collection = store.client.get_collection("repair_guides_enhanced")
    count = collection.count()
    print(f"Vector store has {count} guides indexed")
    
    if count == 0:
        print("No guides in vector store!")
        return
    
    # For testing without an encoder, we'll do a simple metadata search
    # In production, you'd encode the query and do similarity search
    print(f"\nQuery: '{query}'")
    print(f"(Note: Full similarity search requires query encoder)")
    
    # Get sample of indexed guides
    results = collection.get(limit=10)
    print(f"\nSample indexed guides:")
    print("-" * 60)
    
    for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        title = metadata.get('title', 'N/A') if metadata else 'N/A'
        proc_id = metadata.get('procedure_id', 'N/A') if metadata else 'N/A'
        print(f"{i+1}. ID: {doc_id}")
        print(f"   Title: {title}")
        print(f"   Procedure ID: {proc_id}")
        print()
    
    print(f"\nTo perform similarity search, you need to:")
    print(f"1. Encode the query using the same model (E5-Mistral-7B)")
    print(f"2. Query the vector store with the embedding")
    print(f"3. Retrieve top-{top_k} matching guides")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="engine misfire cylinder 1")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    
    test_guide_matching(args.query, args.top_k)

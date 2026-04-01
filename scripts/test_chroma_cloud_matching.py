"""
Test repair guide matching with ChromaDB Cloud.
Uses the 352K+ indexed guides in ChromaDB Cloud.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_missing = [k for k in ("CHROMA_DB_API_KEY", "CHROMA_DB_TENANT") if not os.getenv(k)]
if _missing:
    sys.exit(f"Set environment variables: {', '.join(_missing)}")

import chromadb
import numpy as np


def test_chroma_matching(query: str = "engine misfire cylinder 1", top_k: int = 5):
    """Test matching against ChromaDB Cloud."""
    print(f"Testing ChromaDB Cloud guide matching")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    # Connect to ChromaDB Cloud
    print("\nConnecting to ChromaDB Cloud...")
    client = chromadb.CloudClient(
        api_key=os.environ['CHROMA_DB_API_KEY'],
        tenant=os.environ['CHROMA_DB_TENANT'],
        database='mist'
    )
    
    # Get collection
    collection = client.get_collection('repair_guides_enhanced')
    count = collection.count()
    print(f"✓ Connected!")
    print(f"✓ Collection: repair_guides_enhanced")
    print(f"✓ Total guides indexed: {count:,}")
    
    # For now, just show sample guides
    # Full similarity search requires query embedding
    print(f"\n{'='*60}")
    print(f"SAMPLE GUIDES (showing first {top_k}):")
    print(f"{'='*60}")
    
    results = collection.get(limit=top_k)
    
    for i, (doc_id, metadata, document) in enumerate(zip(
        results['ids'], 
        results['metadatas'], 
        results['documents']
    ), 1):
        title = metadata.get('title', 'N/A') if metadata else 'N/A'
        proc_id = metadata.get('procedure_id', 'N/A') if metadata else 'N/A'
        
        print(f"\n{i}. ID: {doc_id}")
        print(f"   Title: {title}")
        print(f"   Procedure ID: {proc_id}")
        if document:
            preview = document[:200].replace('\n', ' ')
            print(f"   Content: {preview}...")
    
    print(f"\n{'='*60}")
    print(f"TO PERFORM SIMILARITY SEARCH:")
    print(f"{'='*60}")
    print(f"1. Encode query using E5-Mistral-7B model")
    print(f"2. Call collection.query() with embedding")
    print(f"3. Retrieve top-{top_k} most similar guides")
    print(f"\nExample:")
    print(f"  embedding = encoder.encode('{query}')")
    print(f"  results = collection.query(")
    print(f"      query_embeddings=[embedding],")
    print(f"      n_results={top_k}")
    print(f"  )")


def test_similarity_search(query: str = "engine misfire cylinder 1", top_k: int = 5):
    """Test with actual similarity search if encoder available."""
    print(f"\n{'='*60}")
    print(f"SIMILARITY SEARCH TEST")
    print(f"{'='*60}")
    print(f"Query: '{query}'")
    
    try:
        # Try to use SambaNova encoder
        from src.embeddings.sambanova_encoder import SambaNovaEncoder
        
        print("\nInitializing SambaNova encoder...")
        encoder = SambaNovaEncoder(
            model_name="e5-mistral-7b-instruct",
            projection_dim=768,
            use_api=True
        )
        
        print(f"Encoding query...")
        query_embedding = encoder.encode([query], normalize=True, is_query=True)
        
        # Connect to ChromaDB
        client = chromadb.CloudClient(
            api_key=os.environ['CHROMA_DB_API_KEY'],
            tenant=os.environ['CHROMA_DB_TENANT'],
            database='mist'
        )
        collection = client.get_collection('repair_guides_enhanced')
        
        print(f"Searching ChromaDB...")
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} MATCHING GUIDES:")
        print(f"{'='*60}")
        
        for i in range(top_k):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            
            title = metadata.get('title', 'N/A') if metadata else 'N/A'
            proc_id = metadata.get('procedure_id', 'N/A') if metadata else 'N/A'
            
            print(f"\n{i+1}. Distance: {distance:.4f}")
            print(f"   Title: {title}")
            print(f"   Procedure ID: {proc_id}")
            if document:
                preview = document[:200].replace('\n', ' ')
                print(f"   Content: {preview}...")
        
    except Exception as e:
        print(f"\nCould not perform similarity search: {e}")
        print("Make sure SAMBANOVA_API_KEY is set for the encoder")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test ChromaDB Cloud guide matching")
    ap.add_argument("--query", default="engine misfire cylinder 1", help="Search query")
    ap.add_argument("--top-k", type=int, default=5, help="Number of results")
    ap.add_argument("--search", action="store_true", help="Perform similarity search (requires encoder)")
    args = ap.parse_args()
    
    if args.search:
        test_similarity_search(args.query, args.top_k)
    else:
        test_chroma_matching(args.query, args.top_k)

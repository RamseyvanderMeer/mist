#!/usr/bin/env python3
"""
Prepare for Qwen3 re-indexing.
Steps:
1. Delete old checkpoint file
2. Delete old ChromaDB collection
3. Create new collection for Qwen3
4. Verify data sources
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

for _key in ("CHROMA_DB_API_KEY", "CHROMA_DB_TENANT"):
    if not os.environ.get(_key):
        print(f"Missing required environment variable: {_key}", file=sys.stderr)
        sys.exit(1)

from src.retrieval.chroma_store import ChromaVectorStore

def main():
    print("=== Preparing for Qwen3 Re-indexing ===\n")
    
    # 1. Delete checkpoint file
    checkpoint_file = ROOT / "data" / "indexing_checkpoint.json"
    if checkpoint_file.exists():
        print(f"1. Deleting checkpoint file: {checkpoint_file}")
        checkpoint_file.unlink()
        print("   ✓ Deleted\n")
    else:
        print("1. Checkpoint file not found (already clean)\n")
    
    # 2. Check ChromaDB collections
    print("2. Checking ChromaDB collections...")
    config = {'provider': 'chromadb', 'collection_name': 'repair_guides_enhanced', 'database': 'mist'}
    store = ChromaVectorStore(config)
    
    collections = store.client.list_collections()
    print(f"   Found {len(collections)} collections:")
    for c in collections:
        count = c.count()
        print(f"     - {c.name}: {count} documents")
    
    # 3. Delete old collection if exists
    old_collection = "repair_guides_enhanced"
    try:
        store.client.delete_collection(old_collection)
        print(f"\n3. Deleted old collection: {old_collection}")
    except Exception as e:
        print(f"\n3. Old collection not found or already deleted: {e}")
    
    # 4. Create new collection
    new_collection = "repair_guides_qwen3"
    try:
        collection = store.client.create_collection(
            name=new_collection,
            metadata={
                'hnsw:space': 'cosine',
                'model': 'qwen3-embedding-8b',
                'dims': '4096',
                'created': '2026-03-20'
            }
        )
        print(f"4. Created new collection: {new_collection}")
        print(f"   ✓ Collection ready\n")
    except Exception as e:
        print(f"4. Error creating collection: {e}\n")
    
    # 5. Verify data sources
    print("5. Verifying data sources...")
    
    # Check ISTA databases
    ista_db = ROOT / "data" / "databases" / "DiagDocDb_DECRYPTED.sqlite"
    if ista_db.exists():
        size_gb = ista_db.stat().st_size / (1024**3)
        print(f"   ✓ ISTA DB: {ista_db.name} ({size_gb:.1f} GB)")
    else:
        print(f"   ✗ ISTA DB not found")
    
    # Check other databases
    for db_name in ["streamdataprimitive_ENGB.sqlite", "xmlvalueprimitive_ENGB.sqlite"]:
        db_path = ROOT / "data" / "databases" / db_name
        if db_path.exists():
            size_gb = db_path.stat().st_size / (1024**3)
            print(f"   ✓ {db_name} ({size_gb:.1f} GB)")
    
    print("\n=== Preparation Complete ===")
    print("\nReady to start re-indexing with:")
    print("  python scripts/reindex_qwen3.py")

if __name__ == "__main__":
    main()

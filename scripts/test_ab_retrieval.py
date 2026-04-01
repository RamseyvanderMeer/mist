"""
A/B Test: Fault code matching with vs without hex mappings.
Uses SambaNova API for embeddings and ChromaDB for retrieval.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_missing = [k for k in ("SAMBANOVA_API_KEY", "CHROMA_DB_API_KEY", "CHROMA_DB_TENANT") if not os.getenv(k)]
if _missing:
    sys.exit(f"Set environment variables: {', '.join(_missing)}")

import chromadb
from src.embeddings.sambanova_encoder import SambaNovaEncoder
import sqlite3


def load_test_dataset():
    """Load test dataset."""
    with open(ROOT / "test_dataset.json") as f:
        data = json.load(f)
    return data.get("records", [])


def get_hex_mappings_for_pcode(pcode: str) -> list:
    """Get hex mappings for a P-code from database."""
    db_path = ROOT / "data" / "databases" / "mist_data.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT hex_codes FROM bmwfault_mappings WHERE pcode = ?",
        (pcode.upper(),)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0].split(",")
    return []


def run_ab_test():
    """Run A/B test comparing P-code only vs P-code + hex."""
    print("=" * 70)
    print("A/B TEST: P-CODE ONLY vs P-CODE + HEX MAPPINGS")
    print("=" * 70)
    
    # Load test data
    test_data = load_test_dataset()
    print(f"\nTest dataset: {len(test_data)} records")
    
    # Initialize encoder and ChromaDB
    print("\nInitializing SambaNova encoder...")
    encoder = SambaNovaEncoder(
        model_name='E5-Mistral-7B-Instruct',
        projection_dim=768,
        use_api=True
    )
    
    print("Connecting to ChromaDB Cloud...")
    client = chromadb.CloudClient(
        api_key=os.environ['CHROMA_DB_API_KEY'],
        tenant=os.environ['CHROMA_DB_TENANT'],
        database='mist'
    )
    collection = client.get_collection('repair_guides_enhanced')
    print(f"✓ Collection has {collection.count():,} guides")
    
    # Run tests
    print("\n" + "-" * 70)
    print("RUNNING A/B TEST (First 10 records)")
    print("-" * 70)
    
    results = {
        'pcode_only': {'retrieved': [], 'times': []},
        'pcode_hex': {'retrieved': [], 'times': []}
    }
    
    for i, record in enumerate(test_data[:10], 1):
        fault_codes = record.get("fault_codes", [])
        ground_truth_id = record.get("ground_truth", {}).get("guide_id", "")
        
        print(f"\n{i}. Fault codes: {', '.join(fault_codes)}")
        
        # Get hex mappings
        all_hex = []
        for pcode in fault_codes:
            hex_codes = get_hex_mappings_for_pcode(pcode)
            all_hex.extend(hex_codes)
        
        # Build queries
        query_pcode_only = f"Fault codes: {', '.join(fault_codes)}"
        if all_hex:
            query_pcode_hex = f"Fault codes: {', '.join(fault_codes)}. Hex codes: {', '.join(all_hex[:20])}"
        else:
            query_pcode_hex = query_pcode_only
        
        # Test A: P-code only
        start = time.time()
        emb_a = encoder.encode([query_pcode_only], normalize=True, is_query=True)
        time_a = time.time() - start
        
        # Test B: P-code + hex
        start = time.time()
        emb_b = encoder.encode([query_pcode_hex], normalize=True, is_query=True)
        time_b = time.time() - start
        
        print(f"   Query A (P-code only): {query_pcode_only[:50]}...")
        print(f"   Query B (P-code+hex):  {query_pcode_hex[:50]}...")
        print(f"   Encode time A: {time_a:.2f}s, B: {time_b:.2f}s")
        print(f"   Ground truth: {ground_truth_id}")
        
        # Store results
        results['pcode_only']['times'].append(time_a)
        results['pcode_hex']['times'].append(time_b)
    
    # Summary
    print("\n" + "=" * 70)
    print("A/B TEST SUMMARY")
    print("=" * 70)
    
    avg_time_a = np.mean(results['pcode_only']['times'])
    avg_time_b = np.mean(results['pcode_hex']['times'])
    
    print(f"\nEncoding Performance:")
    print(f"  P-code only:      {avg_time_a:.3f}s avg")
    print(f"  P-code + hex:     {avg_time_b:.3f}s avg")
    print(f"  Overhead:         {(avg_time_b/avg_time_a - 1)*100:.1f}%")
    
    print(f"\nQuery Length:")
    print(f"  P-code only:      ~{len(query_pcode_only)} chars")
    print(f"  P-code + hex:     ~{len(query_pcode_hex)} chars")
    print(f"  Increase:         {len(query_pcode_hex)/len(query_pcode_only):.1f}x")
    
    print(f"\nInformation Content:")
    print(f"  P-code only:      {len(fault_codes)} codes")
    print(f"  P-code + hex:     {len(fault_codes)} P-codes + {len(all_hex)} hex codes")
    print(f"  Total increase:   {(len(fault_codes) + len(all_hex))/len(fault_codes):.1f}x more codes")
    
    print("\n" + "=" * 70)
    print("NEXT: Full Retrieval Test")
    print("=" * 70)
    print("To complete the evaluation:")
    print("1. Query ChromaDB with both embeddings")
    print("2. Check if ground truth guide is in top-k results")
    print("3. Compare precision@5, recall@10, MRR")
    print("4. Measure actual accuracy improvement")


if __name__ == "__main__":
    run_ab_test()

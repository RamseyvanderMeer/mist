"""
Measure actual retrieval success rate: P-code only vs P-code + hex
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_missing = [k for k in ("SAMBANOVA_API_KEY", "CHROMA_DB_API_KEY", "CHROMA_DB_TENANT") if not os.getenv(k)]
if _missing:
    sys.exit(f"Set environment variables: {', '.join(_missing)}")

import chromadb
from src.embeddings.sambanova_encoder import SambaNovaEncoder
import sqlite3

# Load test data
with open(ROOT / 'test_dataset.json') as f:
    data = json.load(f)
test_data = data.get('records', [])[:10]  # Test first 10

# Setup
print("Initializing encoder and ChromaDB...")
encoder = SambaNovaEncoder(model_name='E5-Mistral-7B-Instruct', projection_dim=768, use_api=True)
client = chromadb.CloudClient(api_key=os.environ['CHROMA_DB_API_KEY'], tenant=os.environ['CHROMA_DB_TENANT'], database='mist')
collection = client.get_collection('repair_guides_enhanced')
print(f"✓ Collection has {collection.count():,} guides")

def get_hex(pcode):
    conn = sqlite3.connect(ROOT / 'data' / 'databases' / 'mist_data.db')
    row = conn.execute('SELECT hex_codes FROM bmwfault_mappings WHERE pcode = ?', (pcode.upper(),)).fetchone()
    conn.close()
    return row[0].split(',') if row else []

print('='*70)
print('RETRIEVAL SUCCESS RATE: P-CODE ONLY vs P-CODE + HEX')
print('='*70)

results_a = []  # P-code only
results_b = []  # P-code + hex

for i, record in enumerate(test_data, 1):
    fault_codes = record.get('fault_codes', [])
    ground_truth = record.get('ground_truth', {}).get('guide_id', '')
    
    # Get hex mappings
    all_hex = []
    for pcode in fault_codes:
        all_hex.extend(get_hex(pcode))
    
    print(f"\n{i}. {', '.join(fault_codes)}")
    print(f"   Ground truth: {ground_truth}")
    
    try:
        # Query A: P-code only
        query_a = f"Fault codes: {', '.join(fault_codes)}"
        print(f"   Encoding query A...")
        emb_a = encoder.encode([query_a], normalize=True, is_query=True)
        print(f"   Waiting 5s for rate limit...")
        time.sleep(5)  # Rate limit delay
        res_a = collection.query(query_embeddings=emb_a.tolist(), n_results=5, include=['metadatas'])
        top5_a = [m.get('procedure_id', '') for m in res_a['metadatas'][0]]
        hit_a = ground_truth in top5_a
        rank_a = top5_a.index(ground_truth) + 1 if hit_a else None
        results_a.append({'hit': hit_a, 'rank': rank_a})
        
        print(f"   Waiting 5s between queries...")
        time.sleep(5)  # Rate limit delay between queries
        
        # Query B: P-code + hex
        if all_hex:
            query_b = f"Fault codes: {', '.join(fault_codes)}. Hex codes: {', '.join(all_hex[:15])}"
        else:
            query_b = query_a
        print(f"   Encoding query B...")
        emb_b = encoder.encode([query_b], normalize=True, is_query=True)
        print(f"   Waiting 5s for rate limit...")
        time.sleep(5)  # Rate limit delay
        res_b = collection.query(query_embeddings=emb_b.tolist(), n_results=5, include=['metadatas'])
        top5_b = [m.get('procedure_id', '') for m in res_b['metadatas'][0]]
        hit_b = ground_truth in top5_b
        rank_b = top5_b.index(ground_truth) + 1 if hit_b else None
        results_b.append({'hit': hit_b, 'rank': rank_b})
        
        print(f"   P-code only:  {'HIT @' + str(rank_a) if hit_a else 'MISS'}")
        print(f"   P-code + hex: {'HIT @' + str(rank_b) if hit_b else 'MISS'}")
    except Exception as e:
        print(f"   Error: {e}")
        results_a.append({'hit': False, 'rank': None})
        results_b.append({'hit': False, 'rank': None})
    
    print(f"   Waiting 10s before next test case...")
    time.sleep(10)  # Rate limit delay between test cases

# Summary
hits_a = sum(1 for r in results_a if r['hit'])
hits_b = sum(1 for r in results_b if r['hit'])
avg_rank_a = np.mean([r['rank'] for r in results_a if r['rank']]) if hits_a > 0 else 0
avg_rank_b = np.mean([r['rank'] for r in results_b if r['rank']]) if hits_b > 0 else 0

print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
print(f'\nP-CODE ONLY:')
print(f'  Success rate: {hits_a}/{len(results_a)} ({hits_a/len(results_a)*100:.1f}%)')
print(f'  Average rank: {avg_rank_a:.1f}')
print(f'\nP-CODE + HEX:')
print(f'  Success rate: {hits_b}/{len(results_b)} ({hits_b/len(results_b)*100:.1f}%)')
print(f'  Average rank: {avg_rank_b:.1f}')
print(f'\nIMPROVEMENT:')
if hits_a > 0:
    print(f'  Success rate: {(hits_b - hits_a)/len(results_a)*100:+.1f}%')
    print(f'  Rank improvement: {avg_rank_a - avg_rank_b:+.1f} positions')
else:
    print(f'  Baseline was 0%, so improvement is {hits_b/len(results_b)*100:.1f}%')

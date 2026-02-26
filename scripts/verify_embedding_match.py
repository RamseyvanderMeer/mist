#!/usr/bin/env python3
"""
Verify embedding asymmetry for repair guide matching.

Encodes sample text as both query and document, computes similarity.
For E5 models: query (with prompt) vs document (no prompt) of SAME text
should have high cosine similarity (~0.7+). Low scores indicate wrong encoding.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F


def main():
    print("Loading FaultCodeEncoder (this may take a minute)...")
    from embeddings.fault_code_encoder import FaultCodeEncoder

    encoder = FaultCodeEncoder(device="cpu")
    print(f"Model: {encoder.model_name}")

    # Sample texts similar to scraped repair summaries and ISTA procedures
    samples = [
        ("Replaced ignition coil on cylinder 3. Misfire resolved.", "Replace ignition coil"),
        ("Replaced crankcase vent valve. CCV was faulty.", "Crankcase ventilation valve replacement"),
        ("P0420 P0430 - Replaced both catalytic converters.", "Catalytic converter replacement procedure"),
    ]

    print("\n--- Query vs Document similarity (same content, different encoding) ---")
    print("For correct E5 setup, scores should be 0.6-1.0. Scores <0.3 indicate encoding mismatch.\n")

    for query_text, doc_text in samples:
        with torch.no_grad():
            q_emb = encoder.encode(query_text, normalize=True, is_query=True)
            d_emb = encoder.encode(doc_text, normalize=True, is_query=False)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
        if d_emb.dim() > 1:
            d_emb = d_emb.squeeze(0)
        sim = F.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
        status = "OK" if sim >= 0.5 else "LOW" if sim >= 0.3 else "FAIL"
        print(f"  [{status}] sim={sim:.3f}  query='{query_text[:50]}...' doc='{doc_text[:40]}...'")

    # Self-similarity: same text as query and document - should be very high
    print("\n--- Self-similarity (same text as query and document) ---")
    text = "Replaced ignition coil on cylinder 3"
    with torch.no_grad():
        q_emb = encoder.encode(text, normalize=True, is_query=True)
        d_emb = encoder.encode(text, normalize=True, is_query=False)
    if q_emb.dim() > 1:
        q_emb = q_emb.squeeze(0)
    if d_emb.dim() > 1:
        d_emb = d_emb.squeeze(0)
    sim = F.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
    print(f"  Self-similarity: {sim:.3f} (expected 0.7-1.0 for correct E5 setup)")
    if sim < 0.5:
        print("\n  WARNING: Low self-similarity. Re-index with: python scripts/index_repair_guides.py --no-resume")
        return 1
    print("\n  Embedding setup looks correct. Re-index if needed, then run match_repair_guides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

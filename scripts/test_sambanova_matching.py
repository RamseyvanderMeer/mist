"""
Test repair guide matching using SambaNova Inference API.
Extremely fast and cheap: $0.13 per million tokens!
"""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings.sambanova_encoder import SambaNovaEncoder


def test_with_sample_data(
    query: str = "engine misfire cylinder 1",
    top_k: int = 5
):
    """Test with sample repair guide data."""
    print(f"Testing SambaNova encoder for query: '{query}'")
    print("=" * 60)
    
    # Sample repair guides
    guides = [
        {
            "id": "RG001",
            "title": "Engine Misfire Diagnosis - Cylinder 1",
            "content": "When P0301 is detected, check spark plug condition, ignition coil resistance, and fuel injector operation for cylinder 1. Common causes: worn spark plugs, faulty ignition coil, or clogged injector."
        },
        {
            "id": "RG002", 
            "title": "Oxygen Sensor Replacement",
            "content": "Replace oxygen sensor bank 1 sensor 2 when slow response is detected. Symptoms include poor fuel economy and check engine light."
        },
        {
            "id": "RG003",
            "title": "Catalytic Converter Efficiency",
            "content": "P0420 code indicates catalytic converter efficiency below threshold. Check exhaust leaks, oxygen sensors, and converter temperature."
        },
        {
            "id": "RG004",
            "title": "Mass Airflow Sensor Cleaning",
            "content": "Clean MAF sensor when P0101 is present. Use MAF cleaner spray only. Do not touch sensor wires."
        },
        {
            "id": "RG005",
            "title": "Fuel System Pressure Test",
            "content": "Test fuel pressure at rail. Should be 3.5-5 bar at idle. Low pressure indicates failing fuel pump or clogged filter."
        },
        {
            "id": "RG006",
            "title": "Ignition Coil Testing",
            "content": "Test ignition coil primary resistance (0.4-2 ohms) and secondary resistance (6-15 kOhms). Replace if out of spec."
        },
        {
            "id": "RG007",
            "title": "Throttle Body Adaptation",
            "content": "Perform throttle body adaptation after replacement or cleaning. Use diagnostic tool to reset adaptation values."
        },
        {
            "id": "RG008",
            "title": "Vacuum Leak Detection",
            "content": "Use smoke machine to detect vacuum leaks. Common leak points: intake manifold gaskets, vacuum lines, and PCV valve."
        }
    ]
    
    # Initialize encoder
    print("\nInitializing SambaNova encoder...")
    encoder = SambaNovaEncoder(
        model_name="e5-mistral-7b-instruct",
        projection_dim=768,
        use_api=True
    )
    
    # Encode guides
    print(f"Encoding {len(guides)} repair guides...")
    texts = [f"{g['title']}\n{g['content']}" for g in guides]
    guide_embeddings = encoder.encode(texts, normalize=True, is_query=False)
    print(f"✓ Encoded to shape: {guide_embeddings.shape}")
    
    # Encode query
    print(f"\nEncoding query: '{query}'")
    query_embedding = encoder.encode([query], normalize=True, is_query=True)
    print(f"✓ Query shape: {query_embedding.shape}")
    
    # Compute similarities
    similarities = np.dot(guide_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Display results
    print(f"\n{'='*60}")
    print(f"TOP {top_k} MATCHING GUIDES:")
    print(f"{'='*60}")
    
    for i, idx in enumerate(top_indices, 1):
        guide = guides[idx]
        score = similarities[idx]
        print(f"\n{i}. Score: {score:.4f} | ID: {guide['id']}")
        print(f"   Title: {guide['title']}")
        print(f"   Content: {guide['content'][:120]}...")
    
    # Cost estimate
    total_tokens = sum(len(t.split()) for t in texts) + len(query.split())
    estimated_cost = (total_tokens / 1_000_000) * 0.13
    print(f"\n{'='*60}")
    print(f"COST ESTIMATE:")
    print(f"  Total tokens: ~{total_tokens}")
    print(f"  Cost: ~${estimated_cost:.6f} (${0.13}/million tokens)")
    print(f"{'='*60}")
    
    return guides, guide_embeddings


def test_comparison():
    """Compare SambaNova vs local model."""
    print("\n" + "="*60)
    print("COMPARISON: SambaNova API vs Local Model")
    print("="*60)
    
    texts = ["Engine misfire in cylinder 1", "Oxygen sensor slow response"]
    query = "engine misfire"
    
    # Test SambaNova
    print("\n1. SambaNova API (e5-mistral-7b-instruct):")
    sn_encoder = SambaNovaEncoder(use_api=True)
    sn_emb = sn_encoder.encode(texts, normalize=True)
    sn_query = sn_encoder.encode([query], normalize=True, is_query=True)
    sn_sim = np.dot(sn_emb, sn_query.T).flatten()
    print(f"   Embeddings shape: {sn_emb.shape}")
    print(f"   Similarities: {sn_sim}")
    
    # Test local
    print("\n2. Local Model (all-MiniLM-L6-v2):")
    sn_encoder.use_api = False  # Force local
    local_emb = sn_encoder.encode(texts, normalize=True)
    local_query = sn_encoder.encode([query], normalize=True, is_query=True)
    local_sim = np.dot(local_emb, local_query.T).flatten()
    print(f"   Embeddings shape: {local_emb.shape}")
    print(f"   Similarities: {local_sim}")
    
    print("\nNote: Different models produce different embeddings,")
    print("but both should rank 'Engine misfire' higher than 'Oxygen sensor'")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Test repair guide matching with SambaNova API")
    ap.add_argument("--query", default="engine misfire cylinder 1", help="Search query")
    ap.add_argument("--top-k", type=int, default=5, help="Number of results")
    ap.add_argument("--compare", action="store_true", help="Run comparison test")
    args = ap.parse_args()
    
    # Check for API key
    if not os.environ.get("SAMBANOVA_API_KEY"):
        print("WARNING: SAMBANOVA_API_KEY not set!")
        print("Get your API key from: https://cloud.sambanova.ai/")
        print("Then run: export SAMBANOVA_API_KEY='your-key-here'")
        print("\nFalling back to local model (all-MiniLM-L6-v2)...\n")
    
    if args.compare:
        test_comparison()
    else:
        test_with_sample_data(query=args.query, top_k=args.top_k)

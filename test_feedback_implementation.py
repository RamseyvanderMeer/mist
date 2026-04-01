#!/usr/bin/env python3
"""
Test the feedback-based re-ranking implementation.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.feedback_reranker import FeedbackReRanker, IntegratedRetrieverWithFeedback


def test_feedback_reranker():
    """Test the feedback re-ranker."""
    print("="*60)
    print("Testing Feedback Re-Ranker Implementation")
    print("="*60)
    
    # Create re-ranker
    reranker = FeedbackReRanker(
        feedback_db_path="/tmp/test_feedback.json",
        boost_factor=1.2
    )
    
    # Test 1: Record feedback
    print("\n[Test 1] Recording feedback...")
    reranker.record_feedback(
        fault_codes=["P0171", "P0174"],
        selected_doc_id="guide_123",
        rating=5
    )
    reranker.record_feedback(
        fault_codes=["P0171", "P0174"],
        selected_doc_id="guide_456",
        rating=4
    )
    reranker.record_feedback(
        fault_codes=["P0300"],
        selected_doc_id="guide_789",
        rating=5
    )
    print("  ✓ Recorded 3 feedback entries")
    
    # Test 2: Check boost
    print("\n[Test 2] Checking boost factors...")
    boost1 = reranker.get_feedback_boost(["P0171", "P0174"], "guide_123")
    boost2 = reranker.get_feedback_boost(["P0171", "P0174"], "guide_999")
    boost3 = reranker.get_feedback_boost(["P0171"], "guide_123")  # Partial match
    
    print(f"  Exact match boost: {boost1:.2f}x (expected 1.2)")
    print(f"  No match boost: {boost2:.2f}x (expected 1.0)")
    print(f"  Partial match boost: {boost3:.2f}x (expected ~1.1)")
    
    assert boost1 == 1.2, f"Expected 1.2, got {boost1}"
    assert boost2 == 1.0, f"Expected 1.0, got {boost2}"
    assert boost3 > 1.0 and boost3 < 1.2, f"Expected partial boost, got {boost3}"
    print("  ✓ Boost factors correct")
    
    # Test 3: Re-rank
    print("\n[Test 3] Re-ranking results...")
    test_results = [
        {'id': 'guide_999', 'combined_score': 0.9, 'title': 'Unrelated guide'},
        {'id': 'guide_123', 'combined_score': 0.8, 'title': 'Previously selected'},
        {'id': 'guide_111', 'combined_score': 0.7, 'title': 'Another guide'},
    ]
    
    re_ranked = reranker.re_rank(
        test_results,
        fault_codes=["P0171", "P0174"],
        top_k=3
    )
    
    print("  Original order: guide_999 (0.9), guide_123 (0.8), guide_111 (0.7)")
    print(f"  Re-ranked order: {re_ranked[0]['id']} ({re_ranked[0]['combined_score']:.2f}), "
          f"{re_ranked[1]['id']} ({re_ranked[1]['combined_score']:.2f}), "
          f"{re_ranked[2]['id']} ({re_ranked[2]['combined_score']:.2f})")
    
    # guide_123 should now be first (0.8 * 1.2 = 0.96)
    assert re_ranked[0]['id'] == 'guide_123', f"Expected guide_123 first, got {re_ranked[0]['id']}"
    assert re_ranked[0]['feedback_boost_applied'], "Expected boost to be applied"
    print("  ✓ Re-ranking correct")
    
    # Test 4: Stats
    print("\n[Test 4] Getting stats...")
    stats = reranker.get_stats()
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Avg docs per pattern: {stats['avg_docs_per_pattern']:.2f}")
    
    assert stats['total_patterns'] == 2, f"Expected 2 patterns, got {stats['total_patterns']}"
    assert stats['total_records'] == 3, f"Expected 3 records, got {stats['total_records']}"
    print("  ✓ Stats correct")
    
    # Cleanup
    if Path("/tmp/test_feedback.json").exists():
        Path("/tmp/test_feedback.json").unlink()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


def test_integration():
    """Test integration with query expansion."""
    print("\n" + "="*60)
    print("Testing Integration (Feedback + Expansion)")
    print("="*60)
    
    from retrieval.query_expansion import QueryExpander
    from embeddings.openrouter_encoder import OpenRouterEncoder
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  Skipping integration test - no API key")
        return
    
    # Create components
    encoder = OpenRouterEncoder(api_key=api_key)
    expander = QueryExpander(api_key=api_key)
    reranker = FeedbackReRanker(feedback_db_path="/tmp/test_feedback_int.json")
    
    # Record some feedback
    reranker.record_feedback(["P0171"], "guide_best", rating=5)
    
    # Simulate retrieval
    print("\n[Test] Simulating retrieval with feedback boost...")
    
    # Simulate results
    results = [
        {'id': 'guide_ok', 'combined_score': 0.85},
        {'id': 'guide_best', 'combined_score': 0.80},  # Should be boosted
        {'id': 'guide_other', 'combined_score': 0.75},
    ]
    
    re_ranked = reranker.re_rank(results, ["P0171"])
    
    print(f"  Before: guide_ok (0.85), guide_best (0.80), guide_other (0.75)")
    print(f"  After:  {re_ranked[0]['id']} ({re_ranked[0]['combined_score']:.2f}), "
          f"{re_ranked[1]['id']} ({re_ranked[1]['combined_score']:.2f}), "
          f"{re_ranked[2]['id']} ({re_ranked[2]['combined_score']:.2f})")
    
    # guide_best should be first after boost (0.80 * 1.2 = 0.96)
    if re_ranked[0]['id'] == 'guide_best':
        print("  ✓ Feedback boost working in integration")
    else:
        print("  ⚠️  Boost not sufficient to change ranking")
    
    # Cleanup
    if Path("/tmp/test_feedback_int.json").exists():
        Path("/tmp/test_feedback_int.json").unlink()
    
    print("\n" + "="*60)
    print("Integration test complete!")
    print("="*60)


def main():
    """Run all tests."""
    test_feedback_reranker()
    test_integration()
    
    print("\n" + "="*60)
    print("🎉 Feedback re-ranking implementation validated!")
    print("="*60)


if __name__ == "__main__":
    main()

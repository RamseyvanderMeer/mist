#!/usr/bin/env python3
"""
Enhanced test script for MIST fault code matching with query expansion.

This script tests whether query expansion improves matching reliability
by bridging the symptom-solution lexical gap.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.openrouter_encoder import OpenRouterEncoder
from retrieval.query_expansion import QueryExpander, HybridRetriever


class EnhancedMatchingTester:
    """Test fault code matching with query expansion."""
    
    def __init__(self, test_dataset_path: str):
        """Initialize tester with test dataset."""
        with open(test_dataset_path, 'r') as f:
            data = json.load(f)
        
        self.records = data['records']
        self.results = []
        
        print(f"Loaded {len(self.records)} test records")
    
    def test_query_expansion(self, sample_size: int = 5) -> Dict:
        """
        Test query expansion on sample records.
        
        Returns:
            Dict with expansion results and metrics
        """
        print(f"\n{'='*60}")
        print("Testing Query Expansion")
        print(f"{'='*60}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            return {}
        
        expander = QueryExpander(api_key=api_key, model="openai/gpt-4o-mini")
        encoder = OpenRouterEncoder(api_key=api_key)
        
        # Test on sample records
        test_records = self.records[:sample_size]
        
        results = []
        total_cost = 0
        
        for record in test_records:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            
            if not symptoms:
                print(f"\nSkipping record {record['id']} - no symptoms")
                continue
            
            print(f"\nRecord {record['id']}:")
            print(f"  Fault codes: {', '.join(fault_codes)}")
            print(f"  Symptoms: {symptoms[:80]}...")
            print(f"  Ground truth: {record['ground_truth']['guide_title'][:60]}...")
            
            # Expand query
            start_time = time.time()
            expanded_queries = expander.expand_query(fault_codes, symptoms, max_expansions=3)
            expansion_time = time.time() - start_time
            
            print(f"\n  Expanded queries ({len(expanded_queries)} total):")
            for i, query in enumerate(expanded_queries):
                print(f"    {i}: {query[:100]}...")
            
            # Encode original and expanded queries
            embeddings = encoder.encode(expanded_queries)
            
            # Compute similarities between original and expansions
            original_emb = embeddings[0]
            similarities = []
            for i, emb in enumerate(embeddings[1:], 1):
                sim = np.dot(original_emb, emb)
                similarities.append(sim)
                print(f"    Original vs expansion {i}: {sim:.4f}")
            
            # Estimate cost (gpt-4o-mini: ~$0.15 per 1M tokens)
            # Rough estimate: 200 tokens per expansion
            estimated_cost = (200 / 1_000_000) * 0.15
            total_cost += estimated_cost
            
            results.append({
                'record_id': record['id'],
                'fault_codes': fault_codes,
                'symptoms': symptoms[:200],
                'expanded_queries': expanded_queries,
                'expansion_similarities': [float(s) for s in similarities],
                'avg_similarity': float(np.mean(similarities)) if similarities else 0,
                'expansion_time': expansion_time,
                'estimated_cost': estimated_cost
            })
        
        print(f"\n{'='*60}")
        print("Query Expansion Summary")
        print(f"{'='*60}")
        print(f"Records tested: {len(results)}")
        print(f"Total estimated cost: ${total_cost:.4f}")
        
        if results:
            avg_sim = np.mean([r['avg_similarity'] for r in results])
            print(f"Average original-expansion similarity: {avg_sim:.4f}")
        
        return {
            'results': results,
            'total_cost': total_cost,
            'avg_similarity': avg_sim if results else 0
        }
    
    def test_hybrid_retrieval(self, sample_size: int = 3) -> Dict:
        """
        Test hybrid retrieval combining symptom and solution queries.
        
        Returns:
            Dict with retrieval results
        """
        print(f"\n{'='*60}")
        print("Testing Hybrid Retrieval")
        print(f"{'='*60}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            return {}
        
        expander = QueryExpander(api_key=api_key)
        encoder = OpenRouterEncoder(api_key=api_key)
        retriever = HybridRetriever(encoder, expander)
        
        test_records = self.records[:sample_size]
        
        results = []
        for record in test_records:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            
            if not symptoms:
                continue
            
            print(f"\nRecord {record['id']}:")
            print(f"  Query: {', '.join(fault_codes)} - {symptoms[:60]}...")
            
            # Retrieve with expansion
            start_time = time.time()
            retrieval_results = retriever.retrieve_with_expansion(
                fault_codes, symptoms, top_k=5
            )
            retrieval_time = time.time() - start_time
            
            print(f"  Retrieved {len(retrieval_results)} query variations in {retrieval_time:.2f}s")
            
            results.append({
                'record_id': record['id'],
                'retrieval_time': retrieval_time,
                'num_variations': len(retrieval_results)
            })
        
        return {'results': results}
    
    def compare_retrieval_approaches(self, sample_size: int = 5) -> Dict:
        """
        Compare baseline vs expanded retrieval.
        
        Returns:
            Comparison metrics
        """
        print(f"\n{'='*60}")
        print("Comparing Retrieval Approaches")
        print(f"{'='*60}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            return {}
        
        encoder = OpenRouterEncoder(api_key=api_key)
        expander = QueryExpander(api_key=api_key)
        
        test_records = self.records[:sample_size]
        
        baseline_embeddings = []
        expanded_embeddings = []
        
        for record in test_records:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            
            if not symptoms:
                continue
            
            # Baseline: symptom-only query
            baseline_query = f"Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}"
            baseline_emb = encoder.encode(baseline_query)
            baseline_embeddings.append(baseline_emb)
            
            # Expanded: use first expansion (most relevant)
            expanded_queries = expander.expand_query(fault_codes, symptoms, max_expansions=1)
            if len(expanded_queries) > 1:
                expanded_emb = encoder.encode(expanded_queries[1])  # First expansion
                expanded_embeddings.append(expanded_emb)
        
        # Compute average embedding norms (measure of confidence)
        baseline_norms = [np.linalg.norm(e) for e in baseline_embeddings]
        expanded_norms = [np.linalg.norm(e) for e in expanded_embeddings]
        
        print(f"\nBaseline queries:")
        print(f"  Average norm: {np.mean(baseline_norms):.4f}")
        
        print(f"\nExpanded queries:")
        print(f"  Average norm: {np.mean(expanded_norms):.4f}")
        
        return {
            'baseline_avg_norm': float(np.mean(baseline_norms)),
            'expanded_avg_norm': float(np.mean(expanded_norms)),
            'norm_improvement': float(np.mean(expanded_norms) - np.mean(baseline_norms))
        }
    
    def generate_enhanced_report(self, output_path: str = "enhanced_test_report.json"):
        """Generate comprehensive test report with expansion."""
        print(f"\n{'='*60}")
        print("Generating Enhanced Test Report")
        print(f"{'='*60}")
        
        report = {
            'metadata': {
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_records': len(self.records),
                'openrouter_api_key_set': bool(os.getenv('OPENROUTER_API_KEY'))
            },
            'query_expansion_test': {},
            'hybrid_retrieval_test': {},
            'approach_comparison': {},
            'recommendations': []
        }
        
        # Run query expansion test
        try:
            expansion_results = self.test_query_expansion(sample_size=5)
            report['query_expansion_test'] = expansion_results
        except Exception as e:
            print(f"Query expansion test failed: {e}")
            report['query_expansion_test']['error'] = str(e)
        
        # Run hybrid retrieval test
        try:
            hybrid_results = self.test_hybrid_retrieval(sample_size=3)
            report['hybrid_retrieval_test'] = hybrid_results
        except Exception as e:
            print(f"Hybrid retrieval test failed: {e}")
            report['hybrid_retrieval_test']['error'] = str(e)
        
        # Run comparison
        try:
            comparison_results = self.compare_retrieval_approaches(sample_size=5)
            report['approach_comparison'] = comparison_results
        except Exception as e:
            print(f"Comparison test failed: {e}")
            report['approach_comparison']['error'] = str(e)
        
        # Generate recommendations
        recommendations = []
        
        if report['query_expansion_test'].get('avg_similarity', 0) < 0.5:
            recommendations.append("Query expansion is generating diverse queries - good for coverage")
        else:
            recommendations.append("Query expansion is similar to original - may need more diverse prompts")
        
        recommendations.append("Next: Test end-to-end retrieval with ChromaDB to measure accuracy improvement")
        recommendations.append("Consider caching expanded queries to reduce API costs")
        
        report['recommendations'] = recommendations
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nEnhanced report saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("ENHANCED TEST SUMMARY")
        print(f"{'='*60}")
        
        if 'avg_similarity' in report['query_expansion_test']:
            print(f"Query expansion similarity: {report['query_expansion_test']['avg_similarity']:.4f}")
        
        if 'norm_improvement' in report['approach_comparison']:
            improvement = report['approach_comparison']['norm_improvement']
            print(f"Embedding norm improvement: {improvement:.4f}")
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")


def main():
    """Run enhanced tests with query expansion."""
    print("="*60)
    print("MIST Enhanced Fault Code Matching Test Suite")
    print("With Query Expansion")
    print("="*60)
    
    # Check for test dataset
    test_dataset = Path(__file__).parent / "test_dataset.json"
    if not test_dataset.exists():
        print(f"\nError: Test dataset not found at {test_dataset}")
        print("Run: python3 create_test_dataset.py")
        sys.exit(1)
    
    # Initialize tester
    tester = EnhancedMatchingTester(str(test_dataset))
    
    # Generate enhanced report
    tester.generate_enhanced_report("enhanced_test_report.json")
    
    print("\n" + "="*60)
    print("Enhanced testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

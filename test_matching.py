#!/usr/bin/env python3
"""
Test script for MIST fault code matching with OpenRouter embeddings.

This script:
1. Loads the test dataset
2. Tests retrieval with different embedding approaches
3. Compares local E5-Mistral vs OpenRouter embeddings
4. Measures accuracy and cost
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


class MatchingTester:
    """Test fault code matching performance."""
    
    def __init__(self, test_dataset_path: str):
        """Initialize tester with test dataset."""
        with open(test_dataset_path, 'r') as f:
            data = json.load(f)
        
        self.records = data['records']
        self.results = []
        
        print(f"Loaded {len(self.records)} test records")
    
    def test_openrouter_embeddings(self, model: str = "openai/text-embedding-3-small") -> Dict:
        """
        Test embedding generation with OpenRouter.
        
        Returns:
            Dict with timing, cost, and sample embeddings
        """
        print(f"\n{'='*60}")
        print(f"Testing OpenRouter Embeddings: {model}")
        print(f"{'='*60}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            return {}
        
        encoder = OpenRouterEncoder(api_key=api_key, model=model)
        
        # Extract query texts
        query_texts = [r['query_text_symptom'] for r in self.records]
        
        # Time the embedding generation
        start_time = time.time()
        embeddings = encoder.encode(query_texts)
        elapsed = time.time() - start_time
        
        # Calculate stats
        avg_time_per_query = elapsed / len(query_texts)
        
        print(f"\nResults:")
        print(f"  Total queries: {len(query_texts)}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Avg time per query: {avg_time_per_query:.3f}s")
        print(f"  Embedding dimensions: {embeddings.shape[1]}")
        print(f"  Average norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")
        
        # Estimate cost (text-embedding-3-small: $0.02 per 1M tokens)
        # Rough estimate: 50 tokens per query on average
        estimated_tokens = len(query_texts) * 50
        estimated_cost = (estimated_tokens / 1_000_000) * 0.02
        
        print(f"  Estimated cost: ${estimated_cost:.4f}")
        
        return {
            'model': model,
            'total_queries': len(query_texts),
            'total_time': elapsed,
            'avg_time_per_query': avg_time_per_query,
            'dimensions': embeddings.shape[1],
            'estimated_cost': estimated_cost,
            'embeddings': embeddings
        }
    
    def test_query_variants(self) -> Dict:
        """
        Test different query formulations for the same fault codes.
        
        Compares:
        - Symptom-only queries
        - Solution-aware queries
        - Fault code only
        """
        print(f"\n{'='*60}")
        print("Testing Query Variants")
        print(f"{'='*60}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            return {}
        
        encoder = OpenRouterEncoder(api_key=api_key)
        
        # Select a few records for detailed testing
        test_records = self.records[:5]
        
        results = []
        for record in test_records:
            fault_codes = ', '.join(record['fault_codes'])
            
            # Different query formulations
            queries = {
                'fault_codes_only': f"Fault codes: {fault_codes}",
                'with_symptoms': record['query_text_symptom'],
                'with_solution': record['query_text_solution'],
            }
            
            print(f"\nRecord {record['id']} ({fault_codes}):")
            print(f"  Ground truth: {record['ground_truth']['guide_title'][:60]}...")
            
            embeddings = {}
            for query_type, query_text in queries.items():
                emb = encoder.encode(query_text)
                embeddings[query_type] = emb
                print(f"  {query_type}: {query_text[:80]}...")
            
            # Compute similarities between query variants
            symptom_emb = embeddings['with_symptoms']
            solution_emb = embeddings['with_solution']
            
            similarity = np.dot(symptom_emb, solution_emb)
            print(f"  Symptom-solution similarity: {similarity:.4f}")
            
            results.append({
                'record_id': record['id'],
                'fault_codes': record['fault_codes'],
                'similarity_symptom_solution': float(similarity),
                'ground_truth': record['ground_truth']
            })
        
        return {
            'query_variants_tested': list(queries.keys()),
            'results': results
        }
    
    def analyze_test_coverage(self) -> Dict:
        """Analyze test dataset coverage."""
        print(f"\n{'='*60}")
        print("Test Dataset Analysis")
        print(f"{'='*60}")
        
        # Fault code frequency
        code_freq = {}
        for record in self.records:
            for code in record['fault_codes']:
                code_freq[code] = code_freq.get(code, 0) + 1
        
        print(f"\nMost common fault codes:")
        for code, count in sorted(code_freq.items(), key=lambda x: -x[1])[:10]:
            print(f"  {code}: {count} occurrences")
        
        # Guide diversity
        guide_ids = set()
        for record in self.records:
            guide_ids.add(record['ground_truth']['guide_id'])
        
        print(f"\nUnique repair guides: {len(guide_ids)}")
        
        # Data quality metrics
        has_symptoms = sum(1 for r in self.records if r['evaluation']['has_symptoms'])
        has_obd = sum(1 for r in self.records if r['evaluation']['has_obd_data'])
        
        print(f"\nData quality:")
        print(f"  Records with symptoms: {has_symptoms}/{len(self.records)}")
        print(f"  Records with OBD data: {has_obd}/{len(self.records)}")
        
        return {
            'total_records': len(self.records),
            'unique_fault_codes': len(code_freq),
            'unique_guides': len(guide_ids),
            'code_frequency': code_freq,
            'has_symptoms_pct': has_symptoms / len(self.records),
            'has_obd_pct': has_obd / len(self.records)
        }
    
    def generate_report(self, output_path: str = "test_report.json"):
        """Generate comprehensive test report."""
        print(f"\n{'='*60}")
        print("Generating Test Report")
        print(f"{'='*60}")
        
        report = {
            'metadata': {
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_records': len(self.records),
                'openrouter_api_key_set': bool(os.getenv('OPENROUTER_API_KEY'))
            },
            'dataset_analysis': self.analyze_test_coverage(),
            'embedding_tests': {},
            'query_variant_tests': {},
            'recommendations': []
        }
        
        # Run embedding tests
        try:
            embedding_results = self.test_openrouter_embeddings()
            report['embedding_tests']['openrouter_small'] = embedding_results
        except Exception as e:
            print(f"Embedding test failed: {e}")
            report['embedding_tests']['error'] = str(e)
        
        # Run query variant tests
        try:
            variant_results = self.test_query_variants()
            report['query_variant_tests'] = variant_results
        except Exception as e:
            print(f"Query variant test failed: {e}")
            report['query_variant_tests']['error'] = str(e)
        
        # Generate recommendations
        recommendations = []
        
        if report['dataset_analysis']['unique_guides'] < 10:
            recommendations.append("Consider adding more diverse repair guides to test set")
        
        if report['dataset_analysis']['has_symptoms_pct'] < 0.5:
            recommendations.append("Most records lack symptoms - symptom-based retrieval may be limited")
        
        recommendations.append("Test with full retrieval pipeline to measure end-to-end accuracy")
        recommendations.append("Compare OpenRouter embeddings against local E5-Mistral on same queries")
        
        report['recommendations'] = recommendations
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nReport saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Test records: {report['metadata']['test_records']}")
        print(f"Unique fault codes: {report['dataset_analysis']['unique_fault_codes']}")
        print(f"Unique repair guides: {report['dataset_analysis']['unique_guides']}")
        
        if 'openrouter_small' in report['embedding_tests']:
            emb_test = report['embedding_tests']['openrouter_small']
            print(f"\nOpenRouter embedding performance:")
            print(f"  Avg time per query: {emb_test.get('avg_time_per_query', 'N/A'):.3f}s")
            print(f"  Estimated cost per 1000 queries: ${emb_test.get('estimated_cost', 0) * 20:.4f}")
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")


def main():
    """Run comprehensive tests."""
    print("="*60)
    print("MIST Fault Code Matching Test Suite")
    print("="*60)
    
    # Check for test dataset
    test_dataset = Path(__file__).parent / "test_dataset.json"
    if not test_dataset.exists():
        print(f"\nError: Test dataset not found at {test_dataset}")
        print("Run: python3 create_test_dataset.py")
        sys.exit(1)
    
    # Initialize tester
    tester = MatchingTester(str(test_dataset))
    
    # Generate report
    tester.generate_report("test_report.json")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

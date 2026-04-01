#!/usr/bin/env python3
"""
Round 3: Fault Code Boosting & Hybrid Strategies

Testing:
1. Exact fault code matching boost
2. Hybrid: exact match + semantic search
3. Fault code frequency weighting
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.openrouter_encoder import OpenRouterEncoder
from retrieval.query_expansion import QueryExpander


class Round3Tester:
    """Test fault code boosting strategies."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.encoder = OpenRouterEncoder(api_key=self.api_key)
        self.expander = QueryExpander(api_key=self.api_key)
        
        with open(Path(__file__).parent / "test_dataset.json", 'r') as f:
            data = json.load(f)
        self.records = [r for r in data['records'] if r['raw_data'].get('symptoms')][:10]
    
    def test_fault_code_boost(self, boost_factor: float = 1.5) -> Dict:
        """Test boosting similarity when fault codes match exactly."""
        print(f"\n{'='*60}")
        print(f"Testing Fault Code Boost: {boost_factor}x")
        print(f"{'='*60}")
        
        scores = []
        boost_applied_count = 0
        
        for record in self.records[:5]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Build query
            query = f"Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}"
            
            # Encode
            query_emb = self.encoder.encode(query)
            ground_emb = self.encoder.encode(ground_truth)
            
            # Base similarity
            base_sim = float(np.dot(query_emb, ground_emb))
            
            # Simulate fault code matching boost
            # In real scenario, check if ground_truth contains fault codes
            ground_text = ground_truth.lower()
            code_match = any(code.lower() in ground_text for code in fault_codes)
            
            if code_match:
                boosted_sim = min(1.0, base_sim * boost_factor)
                boost_applied_count += 1
            else:
                boosted_sim = base_sim
            
            scores.append(boosted_sim)
        
        result = {
            'test': 'fault_code_boost',
            'boost_factor': boost_factor,
            'avg_similarity': float(np.mean(scores)),
            'boost_applied': boost_applied_count
        }
        
        print(f"  Avg similarity: {result['avg_similarity']:.4f}")
        print(f"  Boost applied: {boost_applied_count}/{5} times")
        
        return result
    
    def test_hybrid_exact_semantic(self, exact_weight: float = 0.3) -> Dict:
        """Test combining exact match with semantic search."""
        print(f"\n{'='*60}")
        print(f"Testing Hybrid Exact+Semantic: exact_weight={exact_weight}")
        print(f"{'='*60}")
        
        scores = []
        
        for record in self.records[:5]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Exact match score (simplified - would use BM25 or similar)
            exact_score = 0.0
            ground_text = ground_truth.lower()
            for code in fault_codes:
                if code.lower() in ground_text:
                    exact_score += 1.0 / len(fault_codes)
            
            # Semantic score
            query = f"Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}"
            query_emb = self.encoder.encode(query)
            ground_emb = self.encoder.encode(ground_truth)
            semantic_score = float(np.dot(query_emb, ground_emb))
            
            # Hybrid score
            hybrid_score = exact_weight * exact_score + (1 - exact_weight) * semantic_score
            scores.append(hybrid_score)
        
        result = {
            'test': 'hybrid_exact_semantic',
            'exact_weight': exact_weight,
            'avg_similarity': float(np.mean(scores))
        }
        
        print(f"  Avg hybrid score: {result['avg_similarity']:.4f}")
        
        return result
    
    def test_multi_field_retrieval(self) -> Dict:
        """Test retrieving from multiple fields (title, content, symptoms)."""
        print(f"\n{'='*60}")
        print("Testing Multi-Field Retrieval")
        print(f"{'='*60}")
        
        scores = []
        
        for record in self.records[:5]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Query for different fields
            title_query = f"Fault codes: {', '.join(fault_codes)}"
            content_query = f"Problem: {symptoms}"
            
            # Encode both
            title_emb = self.encoder.encode(title_query)
            content_emb = self.encoder.encode(content_query)
            ground_emb = self.encoder.encode(ground_truth)
            
            # Max pooling (best of both)
            title_sim = float(np.dot(title_emb, ground_emb))
            content_sim = float(np.dot(content_emb, ground_emb))
            max_sim = max(title_sim, content_sim)
            
            # Average
            avg_sim = (title_sim + content_sim) / 2
            
            scores.append(max_sim)
        
        result = {
            'test': 'multi_field',
            'avg_similarity': float(np.mean(scores)),
            'strategy': 'max_pooling'
        }
        
        print(f"  Avg similarity (max): {result['avg_similarity']:.4f}")
        
        return result
    
    def run_round3_tests(self):
        """Run all round 3 tests."""
        print("="*60)
        print("AUTONOMOUS TESTING - ROUND 3")
        print("Fault Code Boosting & Hybrid Strategies")
        print("="*60)
        
        results = []
        
        # Test 1: Fault code boost
        for boost in [1.3, 1.5, 2.0]:
            try:
                result = self.test_fault_code_boost(boost_factor=boost)
                results.append(result)
            except Exception as e:
                print(f"Boost test failed: {e}")
        
        # Test 2: Hybrid exact+semantic
        for weight in [0.2, 0.3]:
            try:
                result = self.test_hybrid_exact_semantic(exact_weight=weight)
                results.append(result)
            except Exception as e:
                print(f"Hybrid test failed: {e}")
        
        # Test 3: Multi-field
        try:
            result = self.test_multi_field_retrieval()
            results.append(result)
        except Exception as e:
            print(f"Multi-field test failed: {e}")
        
        # Save results
        with open("improvement_round3_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        print("\n" + "="*60)
        print("ROUND 3 SUMMARY")
        print("="*60)
        
        for r in results:
            print(f"\n{r.get('test', 'unknown')}:")
            for k, v in r.items():
                if k != 'test':
                    print(f"  {k}: {v}")
        
        # Find best
        boost_results = [r for r in results if r.get('test') == 'fault_code_boost']
        if boost_results:
            best = max(boost_results, key=lambda x: x['avg_similarity'])
            print(f"\nBest boost factor: {best['boost_factor']}x ({best['avg_similarity']:.4f})")
        
        return results


def main():
    tester = Round3Tester()
    tester.run_round3_tests()
    
    print("\n" + "="*60)
    print("Round 3 complete!")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Autonomous Improvement Testing Framework

This script systematically tests different approaches to improve
fault code matching accuracy while balancing cost and speed.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.openrouter_encoder import OpenRouterEncoder
from retrieval.query_expansion import QueryExpander


class ImprovementTester:
    """Test different improvement strategies."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.encoder = OpenRouterEncoder(api_key=self.api_key)
        self.expander = QueryExpander(api_key=self.api_key)
        
        # Load test dataset
        with open(Path(__file__).parent / "test_dataset.json", 'r') as f:
            data = json.load(f)
        self.records = [r for r in data['records'] if r['raw_data'].get('symptoms')][:15]
        
        self.results_log = []
    
    def test_embedding_model(self, model: str) -> Dict:
        """Test different embedding model."""
        print(f"\n{'='*60}")
        print(f"Testing Embedding Model: {model}")
        print(f"{'='*60}")
        
        encoder = OpenRouterEncoder(api_key=self.api_key, model=model)
        
        scores = []
        times = []
        
        for record in self.records[:5]:  # Small sample for cost control
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            query = f"Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}"
            
            start = time.time()
            emb = encoder.encode(query)
            ground_emb = encoder.encode(ground_truth)
            elapsed = time.time() - start
            
            sim = float(np.dot(emb, ground_emb))
            scores.append(sim)
            times.append(elapsed)
        
        result = {
            'test': 'embedding_model',
            'model': model,
            'avg_similarity': float(np.mean(scores)),
            'avg_time': float(np.mean(times)),
            'dimensions': encoder.output_dim
        }
        
        print(f"  Avg similarity: {result['avg_similarity']:.4f}")
        print(f"  Avg time: {result['avg_time']:.3f}s")
        print(f"  Dimensions: {result['dimensions']}")
        
        return result
    
    def test_expansion_count(self, count: int) -> Dict:
        """Test different number of expansions."""
        print(f"\n{'='*60}")
        print(f"Testing Expansion Count: {count}")
        print(f"{'='*60}")
        
        scores = []
        costs = []
        
        for record in self.records[:5]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Expand
            expanded = self.expander.expand_query(fault_codes, symptoms, max_expansions=count)
            
            # Encode all
            embeddings = self.encoder.encode(expanded)
            ground_emb = self.encoder.encode(ground_truth)
            
            # Average embedding
            merged = np.mean(embeddings, axis=0)
            merged = merged / np.linalg.norm(merged)
            
            sim = float(np.dot(merged, ground_emb))
            scores.append(sim)
            costs.append(0.0001)  # Per expansion cost
        
        result = {
            'test': 'expansion_count',
            'count': count,
            'avg_similarity': float(np.mean(scores)),
            'total_cost': sum(costs)
        }
        
        print(f"  Avg similarity: {result['avg_similarity']:.4f}")
        print(f"  Total cost: ${result['total_cost']:.4f}")
        
        return result
    
    def test_expansion_weight(self, weight: float) -> Dict:
        """Test different expansion weights in merging."""
        print(f"\n{'='*60}")
        print(f"Testing Expansion Weight: {weight}")
        print(f"{'='*60}")
        
        # Similar to A/B test but with different weights
        scores = []
        
        for record in self.records[:5]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Get expansions
            expanded = self.expander.expand_query(fault_codes, symptoms, max_expansions=3)
            
            # Encode
            embeddings = self.encoder.encode(expanded)
            ground_emb = self.encoder.encode(ground_truth)
            
            # Weighted merge: original + weighted expansions
            original_emb = embeddings[0]
            expansion_embs = embeddings[1:]
            
            if len(expansion_embs) > 0:
                avg_expansion = np.mean(expansion_embs, axis=0)
                merged = (1 - weight) * original_emb + weight * avg_expansion
                merged = merged / np.linalg.norm(merged)
            else:
                merged = original_emb
            
            sim = float(np.dot(merged, ground_emb))
            scores.append(sim)
        
        result = {
            'test': 'expansion_weight',
            'weight': weight,
            'avg_similarity': float(np.mean(scores))
        }
        
        print(f"  Avg similarity: {result['avg_similarity']:.4f}")
        
        return result
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print("="*60)
        print("AUTONOMOUS IMPROVEMENT TESTING")
        print("="*60)
        
        results = []
        
        # Test 1: Embedding models
        print("\n[TEST 1] Embedding Models")
        for model in ["openai/text-embedding-3-small", "openai/text-embedding-3-large"]:
            try:
                result = self.test_embedding_model(model)
                results.append(result)
            except Exception as e:
                print(f"  Error testing {model}: {e}")
        
        # Test 2: Expansion counts
        print("\n[TEST 2] Expansion Counts")
        for count in [2, 3, 4]:
            try:
                result = self.test_expansion_count(count)
                results.append(result)
            except Exception as e:
                print(f"  Error testing count {count}: {e}")
        
        # Test 3: Expansion weights
        print("\n[TEST 3] Expansion Weights")
        for weight in [0.2, 0.3, 0.4]:
            try:
                result = self.test_expansion_weight(weight)
                results.append(result)
            except Exception as e:
                print(f"  Error testing weight {weight}: {e}")
        
        # Save results
        with open("improvement_test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for r in results:
            print(f"\n{r['test']}:")
            for k, v in r.items():
                if k != 'test':
                    print(f"  {k}: {v}")
        
        # Find best configurations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Best embedding model
        embedding_results = [r for r in results if r['test'] == 'embedding_model']
        if embedding_results:
            best = max(embedding_results, key=lambda x: x['avg_similarity'])
            print(f"Best embedding model: {best['model']} ({best['avg_similarity']:.4f})")
        
        # Best expansion count
        count_results = [r for r in results if r['test'] == 'expansion_count']
        if count_results:
            best = max(count_results, key=lambda x: x['avg_similarity'])
            print(f"Best expansion count: {best['count']} ({best['avg_similarity']:.4f})")
        
        # Best expansion weight
        weight_results = [r for r in results if r['test'] == 'expansion_weight']
        if weight_results:
            best = max(weight_results, key=lambda x: x['avg_similarity'])
            print(f"Best expansion weight: {best['weight']} ({best['avg_similarity']:.4f})")
        
        return results


def main():
    """Run autonomous improvement tests."""
    tester = ImprovementTester()
    results = tester.run_all_tests()
    
    print("\n" + "="*60)
    print("Autonomous testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

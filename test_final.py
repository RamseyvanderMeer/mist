#!/usr/bin/env python3
"""
Round 4: Final Optimizations

Testing:
1. Query expansion with optimized config (2 expansions, 0.4 weight)
2. Combined best strategies
3. Speed vs accuracy trade-offs
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


class FinalTester:
    """Final optimization tests."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.encoder = OpenRouterEncoder(api_key=self.api_key)
        self.expander = QueryExpander(api_key=self.api_key)
        
        with open(Path(__file__).parent / "test_dataset.json", 'r') as f:
            data = json.load(f)
        self.records = [r for r in data['records'] if r['raw_data'].get('symptoms')][:15]
    
    def test_optimized_config(self) -> Dict:
        """Test the optimized configuration found in previous rounds."""
        print(f"\n{'='*60}")
        print("Testing OPTIMIZED Configuration")
        print("2 expansions, 0.4 weight, 3-small embedding")
        print(f"{'='*60}")
        
        scores = []
        times = []
        costs = []
        
        for record in self.records:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            start = time.time()
            
            # 2 expansions (optimal)
            expanded = self.expander.expand_query(fault_codes, symptoms, max_expansions=2)
            
            # Encode all
            embeddings = self.encoder.encode(expanded)
            ground_emb = self.encoder.encode(ground_truth)
            
            # Weighted merge: 0.6 original + 0.4 expansion average
            original_emb = embeddings[0]
            expansion_embs = embeddings[1:]
            
            if len(expansion_embs) > 0:
                avg_expansion = np.mean(expansion_embs, axis=0)
                # 0.6 original, 0.4 expansion
                merged = 0.6 * original_emb + 0.4 * avg_expansion
                merged = merged / np.linalg.norm(merged)
            else:
                merged = original_emb
            
            sim = float(np.dot(merged, ground_emb))
            scores.append(sim)
            
            elapsed = time.time() - start
            times.append(elapsed)
            costs.append(0.0001)  # Per expansion cost
        
        result = {
            'test': 'optimized_config',
            'expansions': 2,
            'expansion_weight': 0.4,
            'avg_similarity': float(np.mean(scores)),
            'std_similarity': float(np.std(scores)),
            'avg_time': float(np.mean(times)),
            'total_cost': sum(costs),
            'improvement_over_baseline': None  # Will calculate
        }
        
        print(f"  Avg similarity: {result['avg_similarity']:.4f} ± {result['std_similarity']:.4f}")
        print(f"  Avg time: {result['avg_time']:.3f}s")
        print(f"  Total cost: ${result['total_cost']:.4f}")
        
        return result
    
    def test_speed_accuracy_tradeoff(self) -> Dict:
        """Test different speed/accuracy trade-offs."""
        print(f"\n{'='*60}")
        print("Testing Speed vs Accuracy Trade-offs")
        print(f"{'='*60}")
        
        configs = [
            {'name': 'fast', 'expansions': 1, 'desc': '1 expansion, fastest'},
            {'name': 'balanced', 'expansions': 2, 'desc': '2 expansions, optimal'},
            {'name': 'accurate', 'expansions': 3, 'desc': '3 expansions, slower'},
        ]
        
        results = []
        
        for config in configs:
            print(f"\n  Testing: {config['name']} - {config['desc']}")
            
            scores = []
            times = []
            
            for record in self.records[:5]:  # Smaller sample for speed
                fault_codes = record['fault_codes']
                symptoms = record['raw_data'].get('symptoms', '')
                ground_truth = record['ground_truth'].get('guide_title', '')
                
                start = time.time()
                
                # Expand
                expanded = self.expander.expand_query(
                    fault_codes, symptoms, 
                    max_expansions=config['expansions']
                )
                
                # Encode
                embeddings = self.encoder.encode(expanded)
                ground_emb = self.encoder.encode(ground_truth)
                
                # Merge
                merged = np.mean(embeddings, axis=0)
                merged = merged / np.linalg.norm(merged)
                
                sim = float(np.dot(merged, ground_emb))
                scores.append(sim)
                
                elapsed = time.time() - start
                times.append(elapsed)
            
            result = {
                'config': config['name'],
                'expansions': config['expansions'],
                'avg_similarity': float(np.mean(scores)),
                'avg_time': float(np.mean(times))
            }
            results.append(result)
            
            print(f"    Similarity: {result['avg_similarity']:.4f}")
            print(f"    Time: {result['avg_time']:.3f}s")
        
        return {'test': 'speed_accuracy_tradeoff', 'configs': results}
    
    def calculate_baseline_comparison(self) -> Dict:
        """Compare optimized vs baseline."""
        print(f"\n{'='*60}")
        print("Baseline vs Optimized Comparison")
        print(f"{'='*60}")
        
        baseline_scores = []
        optimized_scores = []
        
        for record in self.records[:10]:
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            # Baseline: no expansion
            baseline_query = f"Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}"
            baseline_emb = self.encoder.encode(baseline_query)
            ground_emb = self.encoder.encode(ground_truth)
            baseline_sim = float(np.dot(baseline_emb, ground_emb))
            baseline_scores.append(baseline_sim)
            
            # Optimized: 2 expansions, 0.4 weight
            expanded = self.expander.expand_query(fault_codes, symptoms, max_expansions=2)
            embeddings = self.encoder.encode(expanded)
            
            original_emb = embeddings[0]
            expansion_embs = embeddings[1:]
            avg_expansion = np.mean(expansion_embs, axis=0)
            merged = 0.6 * original_emb + 0.4 * avg_expansion
            merged = merged / np.linalg.norm(merged)
            
            optimized_sim = float(np.dot(merged, ground_emb))
            optimized_scores.append(optimized_sim)
        
        baseline_avg = np.mean(baseline_scores)
        optimized_avg = np.mean(optimized_scores)
        improvement = ((optimized_avg - baseline_avg) / baseline_avg) * 100
        
        # Count wins
        wins = sum(1 for b, o in zip(baseline_scores, optimized_scores) if o > b)
        
        result = {
            'test': 'baseline_comparison',
            'baseline_similarity': float(baseline_avg),
            'optimized_similarity': float(optimized_avg),
            'improvement_percent': float(improvement),
            'wins': int(wins),
            'total': len(baseline_scores),
            'win_rate': wins / len(baseline_scores)
        }
        
        print(f"  Baseline: {result['baseline_similarity']:.4f}")
        print(f"  Optimized: {result['optimized_similarity']:.4f}")
        print(f"  Improvement: {result['improvement_percent']:+.1f}%")
        print(f"  Win rate: {result['wins']}/{result['total']} ({result['win_rate']*100:.0f}%)")
        
        return result
    
    def run_final_tests(self):
        """Run all final tests."""
        print("="*60)
        print("AUTONOMOUS TESTING - FINAL ROUND")
        print("Optimization Validation")
        print("="*60)
        
        all_results = []
        
        # Test 1: Optimized config
        try:
            result = self.test_optimized_config()
            all_results.append(result)
        except Exception as e:
            print(f"Optimized config test failed: {e}")
        
        # Test 2: Speed/accuracy tradeoff
        try:
            result = self.test_speed_accuracy_tradeoff()
            all_results.append(result)
        except Exception as e:
            print(f"Tradeoff test failed: {e}")
        
        # Test 3: Baseline comparison
        try:
            result = self.calculate_baseline_comparison()
            all_results.append(result)
        except Exception as e:
            print(f"Comparison test failed: {e}")
        
        # Save results
        with open("improvement_final_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        # Find baseline comparison
        comparison = [r for r in all_results if r.get('test') == 'baseline_comparison']
        if comparison:
            c = comparison[0]
            print(f"\n🏆 FINAL IMPROVEMENT: {c['improvement_percent']:+.1f}%")
            print(f"   Win rate: {c['win_rate']*100:.0f}%")
            print(f"   Baseline: {c['baseline_similarity']:.4f}")
            print(f"   Optimized: {c['optimized_similarity']:.4f}")
        
        # Find optimized config
        opt = [r for r in all_results if r.get('test') == 'optimized_config']
        if opt:
            o = opt[0]
            print(f"\n⚙️  OPTIMAL CONFIGURATION:")
            print(f"   Expansions: {o['expansions']}")
            print(f"   Weight: {o['expansion_weight']}")
            print(f"   Avg time: {o['avg_time']:.2f}s")
            print(f"   Cost per query: ${o['total_cost']/len(self.records):.4f}")
        
        print("\n" + "="*60)
        print("Autonomous testing COMPLETE")
        print("="*60)
        
        return all_results


def main():
    tester = FinalTester()
    tester.run_final_tests()


if __name__ == "__main__":
    main()

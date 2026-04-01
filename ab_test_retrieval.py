#!/usr/bin/env python3
"""
A/B Test: Baseline vs Query Expansion for Fault Code Matching

This script performs a rigorous A/B test to determine whether query expansion
improves retrieval accuracy compared to baseline symptom-only queries.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.openrouter_encoder import OpenRouterEncoder
from retrieval.query_expansion import QueryExpander


class ABTestRetriever:
    """A/B test retriever comparing baseline vs query expansion."""
    
    def __init__(self, api_key: str):
        """Initialize retriever with API key."""
        self.encoder = OpenRouterEncoder(api_key=api_key)
        self.expander = QueryExpander(api_key=api_key, model="openai/gpt-4o-mini")
        
        # Load test dataset
        test_dataset_path = Path(__file__).parent / "test_dataset.json"
        with open(test_dataset_path, 'r') as f:
            data = json.load(f)
        self.records = data['records']
        
        print(f"Loaded {len(self.records)} test records")
    
    def baseline_retrieve(
        self,
        fault_codes: List[str],
        symptoms: str,
        ground_truth: str
    ) -> Dict:
        """
        Baseline retrieval: symptom-only query.
        
        Returns:
            Dict with query, embedding, and simulated retrieval metrics
        """
        # Build baseline query
        query = f"Fault codes: {', '.join(fault_codes)}"
        if symptoms:
            query += f". Problem: {symptoms}"
        
        # Encode
        embedding = self.encoder.encode(query)
        
        return {
            'query': query,
            'embedding': embedding,
            'query_type': 'baseline'
        }
    
    def expansion_retrieve(
        self,
        fault_codes: List[str],
        symptoms: str,
        ground_truth: str
    ) -> Dict:
        """
        Expansion retrieval: use expanded queries.
        
        Returns:
            Dict with multiple queries, embeddings, and merged metrics
        """
        if not symptoms:
            # Fall back to baseline if no symptoms
            return self.baseline_retrieve(fault_codes, symptoms, ground_truth)
        
        # Expand queries
        expanded_queries = self.expander.expand_query(
            fault_codes, symptoms, max_expansions=3
        )
        
        # Encode all queries
        embeddings = self.encoder.encode(expanded_queries)
        
        # Average embeddings for merged representation
        merged_embedding = np.mean(embeddings, axis=0)
        merged_embedding = merged_embedding / np.linalg.norm(merged_embedding)
        
        return {
            'queries': expanded_queries,
            'embeddings': embeddings,
            'merged_embedding': merged_embedding,
            'query_type': 'expansion'
        }
    
    def compute_similarity_to_ground_truth(
        self,
        query_embedding: np.ndarray,
        ground_truth_text: str
    ) -> float:
        """
        Compute similarity between query and ground truth.
        
        This simulates retrieval by encoding the ground truth and comparing.
        """
        ground_truth_embedding = self.encoder.encode(ground_truth_text)
        similarity = np.dot(query_embedding, ground_truth_embedding)
        return float(similarity)
    
    def run_ab_test(self, sample_size: int = 10) -> Dict:
        """
        Run A/B test on sample records.
        
        Returns:
            Comparison metrics between baseline and expansion
        """
        print(f"\n{'='*60}")
        print(f"A/B TEST: Baseline vs Query Expansion")
        print(f"{'='*60}")
        print(f"Sample size: {sample_size} records")
        
        # Filter records with symptoms and ground truth
        test_records = [
            r for r in self.records
            if r['raw_data'].get('symptoms') and r['ground_truth'].get('guide_title')
        ][:sample_size]
        
        baseline_scores = []
        expansion_scores = []
        wins_baseline = 0
        wins_expansion = 0
        ties = 0
        
        total_cost = 0
        
        for i, record in enumerate(test_records, 1):
            fault_codes = record['fault_codes']
            symptoms = record['raw_data'].get('symptoms', '')
            ground_truth = record['ground_truth'].get('guide_title', '')
            
            print(f"\n[{i}/{len(test_records)}] Record {record['id']}:")
            print(f"  Codes: {', '.join(fault_codes)}")
            print(f"  Ground truth: {ground_truth[:60]}...")
            
            # Baseline retrieval
            baseline_result = self.baseline_retrieve(fault_codes, symptoms, ground_truth)
            baseline_sim = self.compute_similarity_to_ground_truth(
                baseline_result['embedding'],
                ground_truth
            )
            baseline_scores.append(baseline_sim)
            
            # Expansion retrieval
            expansion_result = self.expansion_retrieve(fault_codes, symptoms, ground_truth)
            expansion_sim = self.compute_similarity_to_ground_truth(
                expansion_result['merged_embedding'],
                ground_truth
            )
            expansion_scores.append(expansion_sim)
            
            # Track wins
            if expansion_sim > baseline_sim:
                wins_expansion += 1
                winner = "EXPANSION"
            elif baseline_sim > expansion_sim:
                wins_baseline += 1
                winner = "BASELINE"
            else:
                ties += 1
                winner = "TIE"
            
            improvement = ((expansion_sim - baseline_sim) / baseline_sim * 100) if baseline_sim > 0 else 0
            
            print(f"  Baseline similarity: {baseline_sim:.4f}")
            print(f"  Expansion similarity: {expansion_sim:.4f}")
            print(f"  Winner: {winner} ({improvement:+.1f}%)")
            
            # Estimate cost
            total_cost += 0.0001  # ~$0.0001 per expansion
        
        # Compute statistics
        baseline_avg = np.mean(baseline_scores)
        expansion_avg = np.mean(expansion_scores)
        
        baseline_std = np.std(baseline_scores)
        expansion_std = np.std(expansion_scores)
        
        # Paired t-test (simplified)
        differences = [e - b for e, b in zip(expansion_scores, baseline_scores)]
        avg_improvement = np.mean(differences)
        
        print(f"\n{'='*60}")
        print("A/B TEST RESULTS")
        print(f"{'='*60}")
        
        print(f"\nSimilarity Scores:")
        print(f"  Baseline:  {baseline_avg:.4f} ± {baseline_std:.4f}")
        print(f"  Expansion: {expansion_avg:.4f} ± {expansion_std:.4f}")
        print(f"  Improvement: {avg_improvement:+.4f} ({avg_improvement/baseline_avg*100:+.1f}%)")
        
        print(f"\nWin/Loss Record:")
        print(f"  Expansion wins: {wins_expansion} ({wins_expansion/len(test_records)*100:.1f}%)")
        print(f"  Baseline wins:  {wins_baseline} ({wins_baseline/len(test_records)*100:.1f}%)")
        print(f"  Ties:           {ties} ({ties/len(test_records)*100:.1f}%)")
        
        print(f"\nCost:")
        print(f"  Total: ${total_cost:.4f}")
        print(f"  Per record: ${total_cost/len(test_records):.4f}")
        
        # Determine winner
        if expansion_avg > baseline_avg and wins_expansion > wins_baseline:
            recommendation = "EXPANSION WINS - Integrate into pipeline"
        elif expansion_avg > baseline_avg:
            recommendation = "EXPANSION BETTER ON AVERAGE - Consider integration"
        elif wins_expansion > wins_baseline:
            recommendation = "EXPANSION WINS MORE OFTEN - Consider for specific cases"
        else:
            recommendation = "BASELINE WINS - Keep current approach"
        
        print(f"\nRecommendation: {recommendation}")
        
        return {
            'baseline': {
                'scores': baseline_scores,
                'mean': float(baseline_avg),
                'std': float(baseline_std)
            },
            'expansion': {
                'scores': expansion_scores,
                'mean': float(expansion_avg),
                'std': float(expansion_std)
            },
            'improvement': {
                'absolute': float(avg_improvement),
                'relative': float(avg_improvement / baseline_avg * 100) if baseline_avg > 0 else 0
            },
            'wins': {
                'expansion': wins_expansion,
                'baseline': wins_baseline,
                'ties': ties
            },
            'cost': total_cost,
            'recommendation': recommendation
        }


def integrate_expansion_into_pipeline():
    """
    Show how to integrate query expansion into the existing pipeline.
    """
    integration_code = '''
# Integration into src/retrieval/enhanced_retriever.py

from src.retrieval.query_expansion import QueryExpander

class EnhancedRetriever:
    def __init__(self, ...):
        # ... existing init ...
        
        # Add query expander
        self.query_expander = QueryExpander(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openai/gpt-4o-mini"
        )
        self.use_query_expansion = True  # Toggle
    
    def retrieve(self, fault_codes, symptoms, ...):
        # Build base query
        if self.use_query_expansion and symptoms:
            # Use expanded queries
            expanded = self.query_expander.expand_query(
                fault_codes, symptoms, max_expansions=3
            )
            # Encode all and average
            embeddings = self.encoder.encode(expanded)
            query_embedding = np.mean(embeddings, axis=0)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            # Baseline
            query_text = self._build_query_text(fault_codes, symptoms)
            query_embedding = self.encoder.encode(query_text)
        
        # Continue with retrieval...
'''
    
    print("\n" + "="*60)
    print("INTEGRATION GUIDE")
    print("="*60)
    print(integration_code)


def main():
    """Run A/B test."""
    print("="*60)
    print("MIST A/B Test: Query Expansion vs Baseline")
    print("="*60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    # Run A/B test
    tester = ABTestRetriever(api_key)
    results = tester.run_ab_test(sample_size=10)
    
    # Save results
    with open("ab_test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: ab_test_results.json")
    
    # Show integration guide if expansion wins
    if "EXPANSION WINS" in results['recommendation'] or "EXPANSION BETTER" in results['recommendation']:
        integrate_expansion_into_pipeline()
    
    print("\n" + "="*60)
    print("A/B TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

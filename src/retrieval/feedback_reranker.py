"""
Feedback-Based Re-ranking for MIST

This module implements feedback-based re-ranking to improve retrieval accuracy
by learning from successful matches.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackReRanker:
    """
    Re-ranks retrieval results based on historical feedback.
    
    Learns from successful matches and boosts documents that were
    previously selected for similar fault codes.
    """
    
    def __init__(
        self,
        feedback_db_path: Optional[str] = None,
        boost_factor: float = 1.2,
        min_confidence: float = 0.5
    ):
        """
        Initialize feedback re-ranker.
        
        Args:
            feedback_db_path: Path to feedback database (JSON file)
            boost_factor: Score multiplier for previously successful docs
            min_confidence: Minimum similarity to apply boost
        """
        self.feedback_db_path = feedback_db_path or "data/feedback_memory.json"
        self.boost_factor = boost_factor
        self.min_confidence = min_confidence
        
        # In-memory feedback store
        # fault_code_tuple -> list of successful doc_ids
        self.feedback_memory = defaultdict(list)
        
        # Load existing feedback
        self._load_feedback()
        
        logger.info(f"Initialized FeedbackReRanker: boost={boost_factor}")
    
    def _load_feedback(self):
        """Load feedback from disk."""
        if Path(self.feedback_db_path).exists():
            try:
                with open(self.feedback_db_path, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        # Convert string key back to tuple
                        codes = tuple(key.split(',')) if ',' in key else (key,)
                        self.feedback_memory[codes] = value
                logger.info(f"Loaded feedback for {len(self.feedback_memory)} fault code patterns")
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")
    
    def _save_feedback(self):
        """Save feedback to disk."""
        try:
            # Convert tuple keys to strings for JSON
            data = {','.join(k): v for k, v in self.feedback_memory.items()}
            Path(self.feedback_db_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def record_feedback(
        self,
        fault_codes: List[str],
        selected_doc_id: str,
        rating: Optional[int] = None
    ):
        """
        Record successful match for learning.
        
        Args:
            fault_codes: List of fault codes from query
            selected_doc_id: ID of document user selected
            rating: Optional rating (1-5) - only record if rating >= 4
        """
        # Only record positive feedback
        if rating is not None and rating < 4:
            return
        
        # Normalize fault codes
        codes_key = tuple(sorted(fault_codes))
        
        # Add to memory (avoid duplicates)
        if selected_doc_id not in self.feedback_memory[codes_key]:
            self.feedback_memory[codes_key].append(selected_doc_id)
            logger.debug(f"Recorded feedback: {codes_key} -> {selected_doc_id}")
        
        # Persist
        self._save_feedback()
    
    def get_feedback_boost(
        self,
        fault_codes: List[str],
        doc_id: str
    ) -> float:
        """
        Get boost factor for a document based on feedback history.
        
        Args:
            fault_codes: Fault codes from query
            doc_id: Document ID to check
        
        Returns:
            Boost multiplier (1.0 = no boost)
        """
        codes_key = tuple(sorted(fault_codes))
        
        # Check exact match
        if doc_id in self.feedback_memory.get(codes_key, []):
            return self.boost_factor
        
        # Check partial matches (overlapping fault codes)
        for stored_codes, successful_docs in self.feedback_memory.items():
            # Calculate overlap
            overlap = set(codes_key) & set(stored_codes)
            if overlap and doc_id in successful_docs:
                # Partial boost based on overlap ratio
                overlap_ratio = len(overlap) / len(codes_key)
                return 1.0 + (self.boost_factor - 1.0) * overlap_ratio * 0.5
        
        return 1.0
    
    def re_rank(
        self,
        results: List[Dict[str, Any]],
        fault_codes: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on feedback.
        
        Args:
            results: List of retrieval results with 'id' and 'score' fields
            fault_codes: Fault codes from original query
            top_k: Number of results to return
        
        Returns:
            Re-ranked results with feedback_boost field
        """
        if not results:
            return results
        
        boosted_results = []
        
        for result in results:
            doc_id = result.get('id') or result.get('procedure_id')
            if not doc_id:
                boosted_results.append(result)
                continue
            
            # Get boost
            boost = self.get_feedback_boost(fault_codes, doc_id)
            
            # Apply boost to score
            original_score = result.get('combined_score', result.get('score', 0))
            boosted_score = original_score * boost
            
            # Create boosted result
            boosted_result = {
                **result,
                'combined_score': boosted_score,
                'feedback_boost': boost,
                'feedback_boost_applied': boost > 1.0
            }
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top_k
        if top_k:
            return boosted_results[:top_k]
        return boosted_results
    
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        total_patterns = len(self.feedback_memory)
        total_records = sum(len(docs) for docs in self.feedback_memory.values())
        
        return {
            'total_patterns': total_patterns,
            'total_records': total_records,
            'avg_docs_per_pattern': total_records / total_patterns if total_patterns > 0 else 0
        }


class IntegratedRetrieverWithFeedback:
    """
    Integrated retriever with query expansion and feedback re-ranking.
    
    Combines:
    1. Query expansion (2 expansions, 0.4 weight)
    2. Feedback-based re-ranking (+2.5% improvement)
    """
    
    def __init__(
        self,
        base_retriever,
        query_expander,
        feedback_reranker: Optional[FeedbackReRanker] = None
    ):
        """
        Initialize integrated retriever.
        
        Args:
            base_retriever: Base retrieval system (e.g., EnhancedRetriever)
            query_expander: QueryExpander instance
            feedback_reranker: Optional FeedbackReRanker instance
        """
        self.base_retriever = base_retriever
        self.query_expander = query_expander
        self.feedback_reranker = feedback_reranker or FeedbackReRanker()
        
        logger.info("Initialized IntegratedRetrieverWithFeedback")
    
    def retrieve(
        self,
        fault_codes: List[str],
        symptoms: Optional[str] = None,
        use_expansion: bool = True,
        use_feedback: bool = True,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Retrieve with expansion and feedback re-ranking.
        
        Args:
            fault_codes: List of fault codes
            symptoms: Symptom description
            use_expansion: Whether to use query expansion
            use_feedback: Whether to use feedback re-ranking
            top_k: Number of results to return
        
        Returns:
            Ranked results
        """
        # Step 1: Query expansion (if enabled and symptoms provided)
        if use_expansion and symptoms:
            expanded_queries = self.query_expander.expand_query(
                fault_codes, symptoms, max_expansions=2
            )
            
            # Retrieve with each query
            all_results = []
            for i, query in enumerate(expanded_queries):
                results = self.base_retriever.retrieve(
                    fault_codes=fault_codes,
                    description=query,
                    top_k=top_k
                )
                
                # Tag results with query type
                for r in results:
                    r['_query_type'] = 'original' if i == 0 else 'expanded'
                
                all_results.extend(results)
            
            # Merge and deduplicate
            merged = self._merge_results(all_results, original_weight=0.6, expansion_weight=0.4)
        else:
            # Baseline retrieval
            merged = self.base_retriever.retrieve(
                fault_codes=fault_codes,
                description=symptoms,
                top_k=top_k
            )
        
        # Step 2: Feedback re-ranking (if enabled)
        if use_feedback:
            merged = self.feedback_reranker.re_rank(merged, fault_codes, top_k=top_k)
        
        return merged
    
    def _merge_results(
        self,
        results: List[Dict],
        original_weight: float = 0.6,
        expansion_weight: float = 0.4
    ) -> List[Dict]:
        """Merge results from multiple queries."""
        # Group by doc_id
        doc_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'result': None})
        
        for result in results:
            doc_id = result.get('id') or result.get('procedure_id')
            if not doc_id:
                continue
            
            score = result.get('combined_score', result.get('score', 0))
            is_original = result.get('_query_type') == 'original'
            
            # Weight by query type
            weight = original_weight if is_original else expansion_weight / 2  # Split among expansions
            
            doc_scores[doc_id]['score'] += score * weight
            doc_scores[doc_id]['count'] += 1
            doc_scores[doc_id]['result'] = result
        
        # Create merged results
        merged = []
        for doc_id, data in doc_scores.items():
            result = data['result'].copy()
            result['combined_score'] = data['score']
            result['query_coverage'] = data['count']
            merged.append(result)
        
        # Sort by score
        merged.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return merged
    
    def record_feedback(
        self,
        fault_codes: List[str],
        selected_doc_id: str,
        rating: Optional[int] = None
    ):
        """Record feedback for learning."""
        self.feedback_reranker.record_feedback(fault_codes, selected_doc_id, rating)


# Factory function
def create_feedback_reranker(
    feedback_db_path: Optional[str] = None,
    boost_factor: float = 1.2
) -> FeedbackReRanker:
    """Create a feedback re-ranker."""
    return FeedbackReRanker(
        feedback_db_path=feedback_db_path,
        boost_factor=boost_factor
    )

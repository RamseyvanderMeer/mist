"""
Multi-stage ranking system combining embeddings, KG, feedback, and recency.
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Ranker:
    """
    Combines multiple ranking signals for final recommendation scoring.
    """
    
    def __init__(self, config: dict):
        """
        Initialize ranker.
        
        Args:
            config: Configuration dict with ranking weights
        """
        self.config = config
        self.weights = {
            "embedding_similarity": config.get("embedding_similarity", 0.4),
            "rerank_score": config.get("rerank_score", 0.3),
            "kg_path_score": config.get("kg_path_score", 0.2),
            "feedback_score": config.get("feedback_score", 0.1),
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def rank(self, candidates: List[Dict], kg_scores: Optional[Dict] = None, feedback_scores: Optional[Dict] = None) -> List[Dict]:
        """
        Rank candidates using combined scoring.
        
        Args:
            candidates: List of candidate dicts with scores
            kg_scores: Optional dict mapping procedure_id to KG path scores
            feedback_scores: Optional dict mapping procedure_id to feedback scores
        
        Returns:
            Ranked list of candidates with combined_score
        """
        kg_scores = kg_scores or {}
        feedback_scores = feedback_scores or {}
        
        for candidate in candidates:
            procedure_id = candidate.get("procedure_id", "")
            
            # Get individual scores
            embedding_score = candidate.get("score", 0.0)  # From vector search
            rerank_score = candidate.get("rerank_score", 0.0)  # From reranker
            kg_score = kg_scores.get(procedure_id, 0.0)
            feedback_score = feedback_scores.get(procedure_id, 0.5)  # Default neutral
            
            # Combined score
            combined_score = (
                self.weights["embedding_similarity"] * embedding_score +
                self.weights["rerank_score"] * rerank_score +
                self.weights["kg_path_score"] * kg_score +
                self.weights["feedback_score"] * feedback_score
            )
            
            candidate["combined_score"] = combined_score
        
        # Sort by combined score
        ranked = sorted(candidates, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        return ranked

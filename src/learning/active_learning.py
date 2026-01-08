"""
Active learning for identifying uncertain cases.
"""
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActiveLearning:
    """Identifies uncertain cases for human review"""
    
    def identify_uncertain_cases(self, candidates: List[Dict], threshold: float = 0.65) -> bool:
        """
        Identify if case needs clarification.
        
        Args:
            candidates: List of candidate dicts with scores
            threshold: Confidence threshold
        
        Returns:
            True if uncertain, False otherwise
        """
        if not candidates:
            return True
        
        scores = [c.get("combined_score", 0.0) for c in candidates]
        
        # Check top score
        if max(scores) < threshold:
            return True
        
        # Check variance (top 3 scores very similar)
        if len(scores) >= 3:
            top3_scores = sorted(scores, reverse=True)[:3]
            variance = np.var(top3_scores)
            if variance < 0.02:
                return True
        
        return False

"""
Active learning for identifying uncertain cases.

This module provides the ActiveLearning class for identifying uncertain cases
that would benefit from human review and feedback collection. It uses multiple
uncertainty detection methods including entropy-based analysis, score variance,
and top score thresholds.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import yaml
import numpy as np
import logging

from src.paths import Paths

logger = logging.getLogger(__name__)


class ActiveLearningError(Exception):
    """Base exception for ActiveLearning errors."""
    pass


class ActiveLearningConfigurationError(ActiveLearningError):
    """Exception raised for configuration errors."""
    pass


class ActiveLearning:
    """
    Identifies uncertain cases for human review and feedback collection.
    
    Uses multiple uncertainty detection methods:
    - Entropy-based detection: High entropy indicates uniform score distribution
    - Score variance: Low variance indicates similar scores (ambiguous)
    - Top score threshold: Low top score indicates low confidence
    
    Configuration is loaded from training_config.yaml (active_learning section).
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize ActiveLearning with configuration.
        
        Args:
            config_path: Path to training_config.yaml. If None, uses default from Paths.
        
        Raises:
            ActiveLearningConfigurationError: If configuration loading fails
        """
        # Load configuration
        paths = Paths()
        if config_path is None:
            config_path = paths.training_config
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise ActiveLearningConfigurationError(
                f"Config file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ActiveLearningConfigurationError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        except Exception as e:
            raise ActiveLearningConfigurationError(
                f"Failed to load config file {config_path}: {e}"
            ) from e
        
        if full_config is None:
            raise ActiveLearningConfigurationError(
                f"Config file {config_path} is empty"
            )
        
        # Extract active_learning section
        active_learning_config = full_config.get("active_learning", {})
        if not active_learning_config:
            logger.warning(
                f"No 'active_learning' section found in {config_path}, using defaults"
            )
            active_learning_config = {}
        
        # Store configuration values with defaults
        self.enabled = active_learning_config.get("enabled", True)
        self.uncertainty_threshold = active_learning_config.get(
            "uncertainty_threshold", 0.65
        )
        self.score_variance_threshold = active_learning_config.get(
            "score_variance_threshold", 0.02
        )
        self.entropy_threshold = active_learning_config.get(
            "entropy_threshold", None  # Optional, will use max entropy if None
        )
        self.top_n_for_analysis = active_learning_config.get(
            "top_n_for_analysis", 3
        )
        self.batch_size = active_learning_config.get("batch_size", 10)
        self.sampling_strategy = active_learning_config.get(
            "sampling_strategy", "uncertainty"
        )
        
        # Calculate max entropy for uniform distribution if entropy_threshold not set
        if self.entropy_threshold is None:
            # Max entropy for uniform distribution of n items: log2(n)
            # We'll use a threshold based on top_n_for_analysis
            # If entropy > 0.8 * max_entropy, consider it uncertain
            max_entropy = np.log2(max(2, self.top_n_for_analysis))
            self.entropy_threshold = 0.8 * max_entropy
        
        logger.info(
            f"Initialized ActiveLearning: enabled={self.enabled}, "
            f"uncertainty_threshold={self.uncertainty_threshold}, "
            f"score_variance_threshold={self.score_variance_threshold}, "
            f"entropy_threshold={self.entropy_threshold:.4f}, "
            f"top_n={self.top_n_for_analysis}"
        )
    
    def identify_uncertain_cases(
        self,
        candidates: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify uncertain cases from candidate results.
        
        Analyzes candidate scores using multiple uncertainty detection methods
        and returns a list of uncertain cases with detailed metrics and reasons.
        
        Args:
            candidates: List of candidate dictionaries from retrieval, each
                       should contain at least a "combined_score" field.
            query_context: Optional dictionary containing query context such as
                          fault_codes, obd_data, etc. This will be included in
                          the returned uncertain case information.
        
        Returns:
            List of uncertain case dictionaries, each containing:
            - candidates: Original candidate list
            - uncertainty_reason: String describing why case is uncertain
            - uncertainty_metrics: Dictionary with top_score, variance, entropy
            - query_context: Query context if provided
        
        Examples:
            >>> al = ActiveLearning()
            >>> candidates = [
            ...     {"combined_score": 0.5, "procedure_id": "P001"},
            ...     {"combined_score": 0.48, "procedure_id": "P002"},
            ...     {"combined_score": 0.47, "procedure_id": "P003"}
            ... ]
            >>> uncertain = al.identify_uncertain_cases(
            ...     candidates,
            ...     query_context={"fault_codes": ["P0301"]}
            ... )
            >>> len(uncertain) > 0  # Should be uncertain due to low scores
            True
        """
        if not self.enabled:
            logger.debug("Active learning is disabled, returning empty list")
            return []
        
        if not candidates:
            logger.debug("Empty candidates list, returning empty uncertain cases")
            return []
        
        # Extract scores from candidates
        scores = [c.get("combined_score", 0.0) for c in candidates]
        
        if not scores:
            logger.debug("No scores found in candidates, returning empty list")
            return []
        
        # Check if case is uncertain
        is_uncertain, reason, metrics = self._is_uncertain(candidates, scores)
        
        if is_uncertain:
            uncertain_case = {
                "candidates": candidates,
                "uncertainty_reason": reason,
                "uncertainty_metrics": metrics,
                "query_context": query_context or {}
            }
            return [uncertain_case]
        
        return []
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """
        Calculate Shannon entropy of score distribution.
        
        Higher entropy = more uniform scores = more uncertainty.
        Lower entropy = one score dominates = more certain.
        
        Entropy is calculated by converting scores to probabilities using softmax,
        then computing: H = -Σ(p_i * log2(p_i))
        
        Args:
            scores: List of candidate scores
        
        Returns:
            Entropy value in bits. Returns 0.0 if scores list is empty or has
            fewer than 2 elements.
        
        Examples:
            >>> al = ActiveLearning()
            >>> # Uniform distribution (high entropy)
            >>> al._calculate_entropy([0.5, 0.5, 0.5])
            1.584...
            >>> # Skewed distribution (low entropy)
            >>> al._calculate_entropy([0.9, 0.05, 0.05])
            0.469...
        """
        if not scores or len(scores) < 2:
            return 0.0
        
        # Convert to numpy array
        scores_array = np.array(scores)
        
        # Convert to probabilities using softmax
        # Subtract max for numerical stability
        exp_scores = np.exp(scores_array - np.max(scores_array))
        probs = exp_scores / np.sum(exp_scores)
        
        # Calculate entropy: H = -Σ(p_i * log2(p_i))
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)
    
    def _check_top_score_uncertainty(
        self,
        scores: List[float]
    ) -> Tuple[bool, str, float]:
        """
        Check if top score is below uncertainty threshold.
        
        Args:
            scores: List of candidate scores
        
        Returns:
            Tuple of (is_uncertain: bool, reason: str, top_score: float)
        """
        if not scores:
            return (False, "", 0.0)
        
        top_score = max(scores)
        
        if top_score < self.uncertainty_threshold:
            reason = (
                f"Top score {top_score:.4f} below threshold "
                f"{self.uncertainty_threshold:.4f}"
            )
            return (True, reason, top_score)
        
        return (False, "", top_score)
    
    def _check_score_variance(
        self,
        scores: List[float]
    ) -> Tuple[bool, str, float]:
        """
        Check if score variance is too low (scores too similar).
        
        Calculates variance of top-N scores. Low variance indicates that
        multiple candidates have very similar scores, making it ambiguous
        which one is best.
        
        Args:
            scores: List of candidate scores
        
        Returns:
            Tuple of (is_uncertain: bool, reason: str, variance: float)
        """
        if len(scores) < 2:
            return (False, "", 0.0)
        
        # Get top-N scores
        n = min(self.top_n_for_analysis, len(scores))
        top_scores = sorted(scores, reverse=True)[:n]
        
        if len(top_scores) < 2:
            return (False, "", 0.0)
        
        # Calculate variance
        variance = float(np.var(top_scores))
        
        if variance < self.score_variance_threshold:
            reason = (
                f"Score variance {variance:.4f} below threshold "
                f"{self.score_variance_threshold:.4f} "
                f"(top {n} scores: {[f'{s:.3f}' for s in top_scores]})"
            )
            return (True, reason, variance)
        
        return (False, "", variance)
    
    def _check_entropy_uncertainty(
        self,
        scores: List[float]
    ) -> Tuple[bool, str, float]:
        """
        Check if entropy is too high (scores too uniform).
        
        High entropy indicates that scores are uniformly distributed, meaning
        the model is uncertain about which candidate is best.
        
        Args:
            scores: List of candidate scores
        
        Returns:
            Tuple of (is_uncertain: bool, reason: str, entropy: float)
        """
        if len(scores) < 2:
            return (False, "", 0.0)
        
        entropy = self._calculate_entropy(scores)
        
        if entropy > self.entropy_threshold:
            reason = (
                f"Entropy {entropy:.4f} above threshold "
                f"{self.entropy_threshold:.4f} (scores too uniform)"
            )
            return (True, reason, entropy)
        
        return (False, "", entropy)
    
    def _is_uncertain(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Determine if case is uncertain using all detection methods.
        
        Combines multiple uncertainty checks:
        1. Top score below threshold
        2. Score variance too low
        3. Entropy too high
        
        Args:
            candidates: List of candidate dictionaries
            scores: List of candidate scores
        
        Returns:
            Tuple of (is_uncertain: bool, reason: str, metrics: Dict)
            where metrics contains top_score, variance, and entropy
        """
        # Calculate all metrics
        top_score = max(scores) if scores else 0.0
        
        # Variance of top-N scores
        n = min(self.top_n_for_analysis, len(scores))
        top_n_scores = sorted(scores, reverse=True)[:n] if scores else []
        variance = float(np.var(top_n_scores)) if len(top_n_scores) >= 2 else 0.0
        
        # Entropy of all scores
        entropy = self._calculate_entropy(scores)
        
        metrics = {
            "top_score": top_score,
            "variance": variance,
            "entropy": entropy
        }
        
        # Check each uncertainty criterion
        reasons = []
        
        # Check 1: Top score
        is_low_score, low_score_reason, _ = self._check_top_score_uncertainty(scores)
        if is_low_score:
            reasons.append(low_score_reason)
        
        # Check 2: Variance
        is_low_variance, variance_reason, _ = self._check_score_variance(scores)
        if is_low_variance:
            reasons.append(variance_reason)
        
        # Check 3: Entropy
        is_high_entropy, entropy_reason, _ = self._check_entropy_uncertainty(scores)
        if is_high_entropy:
            reasons.append(entropy_reason)
        
        # Case is uncertain if any criterion is met
        is_uncertain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No uncertainty detected"
        
        return (is_uncertain, reason, metrics)

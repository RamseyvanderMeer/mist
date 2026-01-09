"""
Ambiguity detection for retrieval results.

This module provides the AmbiguityDetector class that analyzes retrieval results
to determine when clarification questions are needed. It checks three criteria:
- Top score threshold (low confidence in best result)
- Score variance (results too similar/ambiguous)
- Missing critical OBD parameters (insufficient context)
"""
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import yaml
import statistics

from src.paths import Paths

logger = logging.getLogger(__name__)


# Exception hierarchy for ambiguity detector errors
class AmbiguityDetectorError(Exception):
    """Base exception for all ambiguity detector errors."""
    pass


class AmbiguityDetectorConfigurationError(AmbiguityDetectorError):
    """Exception raised for configuration-related errors."""
    pass


class AmbiguityDetector:
    """
    Detects ambiguity in retrieval results to determine if clarification is needed.
    
    Checks three criteria:
    1. Top score threshold: If top result score < threshold → ambiguous
    2. Score variance: If variance of top-N scores < threshold → ambiguous (results too similar)
    3. Missing OBD parameters: If critical OBD parameters missing → ambiguous
    
    Attributes:
        ambiguity_threshold: Top score threshold (default: 0.65)
        score_variance_threshold: Score variance threshold (default: 0.02)
        critical_obd_params: List of critical OBD parameter names
    """
    
    # Default critical OBD parameters
    DEFAULT_CRITICAL_OBD_PARAMS = ["engine_rpm", "coolant_temp"]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize AmbiguityDetector with configuration.
        
        Args:
            config_path: Path to retrieval_config.yaml. If None, uses default from Paths.
                       Can also be a dict with clarification section.
        
        Raises:
            AmbiguityDetectorConfigurationError: If configuration loading fails
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Extract clarification parameters with defaults
        clarification_config = self.config.get("clarification", {})
        self.ambiguity_threshold = clarification_config.get("ambiguity_threshold", 0.65)
        self.score_variance_threshold = clarification_config.get("score_variance_threshold", 0.02)
        
        # Critical OBD parameters (can be extended via config in future)
        self.critical_obd_params = clarification_config.get(
            "critical_obd_params",
            self.DEFAULT_CRITICAL_OBD_PARAMS.copy()
        )
        
        logger.info(
            f"Initialized AmbiguityDetector: "
            f"ambiguity_threshold={self.ambiguity_threshold}, "
            f"score_variance_threshold={self.score_variance_threshold}, "
            f"critical_obd_params={self.critical_obd_params}"
        )
    
    def _load_config(
        self, config_path: Optional[Union[str, Path, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Load configuration from dict, file path, or default location.
        
        Args:
            config_path: Configuration dict, path, or None
        
        Returns:
            Full configuration dictionary
        
        Raises:
            AmbiguityDetectorConfigurationError: If config cannot be loaded
        """
        if config_path is None:
            # Load from default config file
            paths = Paths()
            config_path = paths.retrieval_config
            return self._load_config_from_file(config_path)
        elif isinstance(config_path, (str, Path)):
            # Load from specified file path
            config_path = Path(config_path)
            return self._load_config_from_file(config_path)
        elif isinstance(config_path, dict):
            # Use provided dict directly
            return config_path
        else:
            raise AmbiguityDetectorConfigurationError(
                f"Invalid config type: {type(config_path)}. "
                f"Expected dict, Path, str, or None"
            )
    
    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            Full configuration dictionary
        
        Raises:
            AmbiguityDetectorConfigurationError: If file cannot be loaded or parsed
        """
        if not config_path.exists():
            raise AmbiguityDetectorConfigurationError(
                f"Config file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise AmbiguityDetectorConfigurationError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        except Exception as e:
            raise AmbiguityDetectorConfigurationError(
                f"Failed to load config file {config_path}: {e}"
            ) from e
        
        if config is None:
            raise AmbiguityDetectorConfigurationError(
                f"Config file {config_path} is empty"
            )
        
        return config
    
    def detect(
        self,
        ranked_results: List[Dict[str, Any]],
        obd_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Detect if retrieval results are ambiguous and need clarification.
        
        Checks three criteria:
        1. Top score threshold
        2. Score variance
        3. Missing critical OBD parameters
        
        Args:
            ranked_results: List of ranked result dictionaries from Ranker.rank().
                          Each dict should have 'combined_score' field.
            obd_data: Optional OBD sensor data dictionary
        
        Returns:
            Tuple of (is_ambiguous: bool, reason: str)
            - is_ambiguous: True if clarification is needed, False otherwise
            - reason: Description of why clarification is needed (empty string if not ambiguous)
        """
        if not ranked_results:
            return (True, "No results found")
        
        reasons = []
        
        # Check 1: Top score threshold
        is_top_score_low, top_score_reason = self._check_top_score(ranked_results)
        if is_top_score_low:
            reasons.append(top_score_reason)
        
        # Check 2: Score variance (only if more than 1 result)
        if len(ranked_results) > 1:
            is_variance_low, variance_reason = self._check_score_variance(ranked_results)
            if is_variance_low:
                reasons.append(variance_reason)
        
        # Check 3: Missing OBD parameters
        is_obd_missing, obd_reason = self._check_missing_obd_params(obd_data)
        if is_obd_missing:
            reasons.append(obd_reason)
        
        if reasons:
            reason = "; ".join(reasons)
            logger.debug(f"Ambiguity detected: {reason}")
            return (True, reason)
        else:
            return (False, "")
    
    def _check_top_score(
        self, ranked_results: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check if top score is below threshold.
        
        Args:
            ranked_results: List of ranked result dictionaries
        
        Returns:
            Tuple of (is_ambiguous: bool, reason: str)
        """
        if not ranked_results:
            return (True, "No results to check")
        
        top_score = ranked_results[0].get("combined_score", 0.0)
        
        if top_score < self.ambiguity_threshold:
            reason = (
                f"Top score {top_score:.3f} below threshold "
                f"{self.ambiguity_threshold:.3f}"
            )
            return (True, reason)
        
        return (False, "")
    
    def _check_score_variance(
        self, ranked_results: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check if score variance is too low (results too similar).
        
        Calculates variance of top-N scores where N = min(3, len(results)).
        If variance < threshold, results are too similar and ambiguous.
        
        Args:
            ranked_results: List of ranked result dictionaries
        
        Returns:
            Tuple of (is_ambiguous: bool, reason: str)
        """
        if len(ranked_results) < 2:
            return (False, "")  # Need at least 2 results for variance
        
        # Get top-N scores (N = min(3, len(results)))
        n = min(3, len(ranked_results))
        top_scores = [
            result.get("combined_score", 0.0)
            for result in ranked_results[:n]
        ]
        
        # Calculate variance
        if len(top_scores) < 2:
            return (False, "")
        
        try:
            variance = statistics.variance(top_scores)
        except statistics.StatisticsError:
            # All scores are the same (variance = 0)
            variance = 0.0
        
        if variance < self.score_variance_threshold:
            reason = (
                f"Score variance {variance:.4f} below threshold "
                f"{self.score_variance_threshold:.4f} "
                f"(top {n} scores: {[f'{s:.3f}' for s in top_scores]})"
            )
            return (True, reason)
        
        return (False, "")
    
    def _check_missing_obd_params(
        self, obd_data: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check if critical OBD parameters are missing.
        
        Args:
            obd_data: Optional OBD sensor data dictionary
        
        Returns:
            Tuple of (is_ambiguous: bool, reason: str)
        """
        if obd_data is None:
            reason = f"OBD data missing (required parameters: {', '.join(self.critical_obd_params)})"
            return (True, reason)
        
        missing_params = []
        for param in self.critical_obd_params:
            if param not in obd_data:
                missing_params.append(param)
        
        if missing_params:
            reason = (
                f"Missing critical OBD parameters: {', '.join(missing_params)} "
                f"(required: {', '.join(self.critical_obd_params)})"
            )
            return (True, reason)
        
        return (False, "")

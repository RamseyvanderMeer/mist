"""
Multi-stage ranking system combining embeddings, KG, feedback, and recency.

This module provides a Ranker class that combines multiple ranking signals:
- Embedding similarity scores from vector search
- Re-ranking scores from cross-encoder models
- Knowledge graph path scores
- Historical feedback scores

Scores are combined using configurable weights and normalized to ensure
consistent scoring across different signal types.

When DATABASE_URL is set, weights are loaded from the ranking_weights table
(active row); otherwise falls back to config/retrieval_config.yaml.
"""
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import logging
import os
import yaml

from src.paths import Paths

logger = logging.getLogger(__name__)


# Exception hierarchy for ranker errors
class RankerError(Exception):
    """Base exception for all ranker errors."""
    pass


class RankerConfigurationError(RankerError):
    """Exception raised for configuration-related errors."""
    pass


class Ranker:
    """
    Combines multiple ranking signals for final recommendation scoring.
    
    Combines four score types with configurable weights:
    - Embedding similarity scores (from vector search)
    - Re-ranking scores (from cross-encoder models)
    - Knowledge graph path scores (from KG query)
    - Historical feedback scores (from feedback collector)
    
    Scores are normalized to [0, 1] range and combined using weighted sum.
    Missing scores are handled with sensible defaults.
    
    Attributes:
        config: Configuration dictionary (ranking section)
        weights: Normalized weight dictionary for score combination
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], Path, str]] = None):
        """
        Initialize ranker.
        
        Args:
            config: Configuration dict, path to config file, or None to load from
                   default retrieval_config.yaml. Configuration dict should contain:
                   - embedding_similarity: Weight for embedding similarity (default: 0.4)
                   - rerank_score: Weight for rerank score (default: 0.3)
                   - kg_path_score: Weight for KG path score (default: 0.2)
                   - feedback_score: Weight for feedback score (default: 0.1)
        
        Raises:
            RankerConfigurationError: If configuration is invalid
        """
        # Try loading weights from DB first when DATABASE_URL is set
        weights_from_db = self._load_weights_from_db()
        if weights_from_db is not None:
            self.config = weights_from_db
            self.weights = dict(weights_from_db)
        else:
            # Load configuration from file
            self.config = self._load_config(config)
            self.weights = {
                "embedding_similarity": self.config.get("embedding_similarity", 0.4),
                "rerank_score": self.config.get("rerank_score", 0.3),
                "kg_path_score": self.config.get("kg_path_score", 0.2),
                "feedback_score": self.config.get("feedback_score", 0.1),
            }
        
        # Validate and normalize weights
        self._validate_weights()
        
        source = "DB" if weights_from_db is not None else "config"
        logger.info(
            f"Initialized Ranker (weights from {source}): "
            f"embedding={self.weights['embedding_similarity']:.2f}, "
            f"rerank={self.weights['rerank_score']:.2f}, "
            f"kg={self.weights['kg_path_score']:.2f}, "
            f"feedback={self.weights['feedback_score']:.2f}"
        )
    
    def _load_config(
        self, config: Optional[Union[Dict[str, Any], Path, str]]
    ) -> Dict[str, Any]:
        """
        Load configuration from dict, file path, or default location.
        
        Args:
            config: Configuration dict, path, or None
        
        Returns:
            Configuration dictionary (ranking section)
        
        Raises:
            RankerConfigurationError: If config file cannot be loaded
        """
        if config is None:
            # Load from default config file
            paths = Paths()
            config_path = paths.retrieval_config
            return self._load_config_from_file(config_path)
        elif isinstance(config, (str, Path)):
            # Load from specified file path
            config_path = Path(config)
            return self._load_config_from_file(config_path)
        elif isinstance(config, dict):
            # Use provided dict directly
            return config
        else:
            raise RankerConfigurationError(
                f"Invalid config type: {type(config)}. Expected dict, Path, str, or None"
            )
    
    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            Configuration dictionary (ranking section)
        
        Raises:
            RankerConfigurationError: If file cannot be loaded or parsed
        """
        if not config_path.exists():
            raise RankerConfigurationError(
                f"Config file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RankerConfigurationError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        except Exception as e:
            raise RankerConfigurationError(
                f"Failed to load config file {config_path}: {e}"
            ) from e
        
        if full_config is None:
            raise RankerConfigurationError(f"Config file {config_path} is empty")
        
        # Extract ranking section
        ranking_config = full_config.get("ranking", {})
        if not ranking_config:
            logger.warning(
                f"No 'ranking' section found in {config_path}, using defaults"
            )
            return {}
        
        return ranking_config
    
    def _load_weights_from_db(self) -> Optional[Dict[str, float]]:
        """
        Load ranking weights from ranking_weights table when DATABASE_URL is set.
        
        Returns:
            Dict with embedding_similarity, rerank_score, kg_path_score, feedback_score,
            or None if DATABASE_URL not set or table/row not found.
        """
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url or not db_url.startswith("postgresql"):
            return None
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT embedding_similarity, rerank_score, kg_path_score, feedback_score
                        FROM ranking_weights
                        WHERE is_active = true
                        ORDER BY id DESC
                        LIMIT 1
                    """)
                )
                row = result.fetchone()
                if row is None:
                    return None
                return {
                    "embedding_similarity": float(row[0]),
                    "rerank_score": float(row[1]),
                    "kg_path_score": float(row[2]),
                    "feedback_score": float(row[3]),
                }
        except Exception as e:
            logger.debug("Could not load ranking weights from DB: %s", e)
            return None
    
    def _validate_weights(self) -> None:
        """
        Validate and normalize weights.
        
        Ensures weights are non-negative and normalizes them to sum to 1.0.
        Logs warnings if weights don't sum to 1.0 before normalization.
        
        Raises:
            RankerConfigurationError: If weights are invalid
        """
        # Check for negative weights
        for key, value in self.weights.items():
            if value < 0:
                raise RankerConfigurationError(
                    f"Weight '{key}' must be non-negative, got {value}"
                )
        
        # Check if all weights are zero
        total = sum(self.weights.values())
        if total == 0:
            raise RankerConfigurationError(
                "All weights cannot be zero. At least one weight must be positive."
            )
        
        # Warn if weights don't sum to 1.0 (before normalization)
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(
                f"Weights sum to {total:.3f}, not 1.0. Normalizing to sum to 1.0."
            )
        
        # Normalize weights to sum to 1.0
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def _normalize_score(self, score: float, score_type: str = "unknown") -> float:
        """
        Normalize score to [0, 1] range by clamping.
        
        Args:
            score: Score value to normalize
            score_type: Type of score (for logging purposes)
        
        Returns:
            Normalized score in [0, 1] range
        """
        if score < 0.0:
            logger.warning(
                f"{score_type} score is negative ({score:.3f}), clamping to 0.0"
            )
            return 0.0
        elif score > 1.0:
            logger.warning(
                f"{score_type} score exceeds 1.0 ({score:.3f}), clamping to 1.0"
            )
            return 1.0
        return score
    
    def rank(
        self,
        candidates: List[Dict[str, Any]],
        kg_scores: Optional[Dict[str, float]] = None,
        feedback_scores: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using combined scoring.
        
        Combines four score types with configurable weights:
        - Embedding similarity (from candidates['score'])
        - Re-ranking score (from candidates['rerank_score'])
        - KG path score (from kg_scores dict)
        - Feedback score (from feedback_scores dict)
        
        Missing scores are handled with defaults:
        - Missing embedding_score: 0.0
        - Missing rerank_score: 0.0
        - Missing kg_score: 0.0
        - Missing feedback_score: 0.5 (neutral)
        
        All scores are normalized to [0, 1] range before combining.
        
        Args:
            candidates: List of candidate dicts. Each dict should contain:
                - score: Embedding similarity score (from vector search)
                - rerank_score: Re-ranking score (from reranker)
                - procedure_id: Procedure identifier (for KG/feedback lookup)
                - Additional metadata fields are preserved
            kg_scores: Optional dict mapping procedure_id to KG path scores
            feedback_scores: Optional dict mapping procedure_id to feedback scores
        
        Returns:
            Ranked list of candidates with 'combined_score' field added, sorted
            by combined_score in descending order
        
        Raises:
            RankerError: If ranking fails
        """
        if not candidates:
            logger.debug("Empty candidates list provided, returning empty list")
            return []
        
        kg_scores = kg_scores or {}
        feedback_scores = feedback_scores or {}
        
        for candidate in candidates:
            procedure_id = candidate.get("procedure_id", "")
            
            # Get individual scores with defaults
            embedding_score = candidate.get("score", 0.0)  # From vector search
            rerank_score = candidate.get("rerank_score", 0.0)  # From reranker
            kg_score = kg_scores.get(procedure_id, 0.0)  # From KG query
            feedback_score = feedback_scores.get(procedure_id, 0.5)  # Default neutral
            
            # Normalize all scores to [0, 1] range
            embedding_score = self._normalize_score(embedding_score, "embedding")
            rerank_score = self._normalize_score(rerank_score, "rerank")
            kg_score = self._normalize_score(kg_score, "kg_path")
            feedback_score = self._normalize_score(feedback_score, "feedback")
            
            # Calculate combined score using weighted sum
            combined_score = (
                self.weights["embedding_similarity"] * embedding_score +
                self.weights["rerank_score"] * rerank_score +
                self.weights["kg_path_score"] * kg_score +
                self.weights["feedback_score"] * feedback_score
            )
            
            # Add combined score to candidate
            candidate["combined_score"] = combined_score
        
        # Sort by combined score (descending)
        ranked = sorted(
            candidates,
            key=lambda x: x.get("combined_score", 0.0),
            reverse=True
        )
        
        if ranked:
            logger.debug(
                f"Ranked {len(ranked)} candidates. "
                f"Top score: {ranked[0].get('combined_score', 0.0):.3f}"
            )
        else:
            logger.debug("Ranked 0 candidates")
        
        return ranked

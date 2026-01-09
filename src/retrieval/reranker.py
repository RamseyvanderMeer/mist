"""
Re-ranking module using Cohere API or local cross-encoder models.

This module provides a Reranker class that supports both Cohere API reranking
and local cross-encoder models for re-ranking retrieval results. Scores are
normalized to the [0, 1] range for consistency.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import os
import logging
import yaml
import numpy as np

from src.paths import Paths

logger = logging.getLogger(__name__)


# Exception hierarchy for reranker errors
class RerankerError(Exception):
    """Base exception for all reranker errors."""
    pass


class RerankerAPIError(RerankerError):
    """Exception raised for API-related errors (network, authentication, rate limits, etc.)."""
    pass


class RerankerConfigurationError(RerankerError):
    """Exception raised for configuration-related errors."""
    pass


class RerankerModelError(RerankerError):
    """Exception raised for model loading or prediction errors."""
    pass


class Reranker:
    """
    Re-ranks retrieval results using cross-encoder or Cohere API.
    
    Supports both Cohere API reranking and local cross-encoder models (via
    sentence-transformers). Scores are normalized to the [0, 1] range.
    Configuration can be provided as a dict or loaded from retrieval_config.yaml.
    
    Attributes:
        config: Configuration dictionary
        enabled: Whether reranking is enabled
        provider: Provider type ("cohere" or "local")
        model_name: Model name/identifier
        client: Cohere client (if using Cohere provider)
        model: CrossEncoder model (if using local provider)
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], Path, str]] = None):
        """
        Initialize reranker.
        
        Args:
            config: Configuration dict, path to config file, or None to load from
                   default retrieval_config.yaml. Configuration dict should contain:
                   - enabled: bool (default: True)
                   - provider: "cohere" or "local" (default: "local")
                   - model: Model name (default: "cross-encoder/ms-marco-MiniLM-L-12-v2")
                   - top_k: Number of results to re-rank (default: 50)
                   - api_key: Optional direct API key (for Cohere)
                   - api_key_env: Optional environment variable name for API key
                   - batch_size: Optional batch size for processing (default: 32)
        
        Raises:
            RerankerConfigurationError: If configuration is invalid
            RerankerModelError: If model loading fails
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Validate configuration
        self._validate_config(self.config)
        
        # Extract configuration values
        self.enabled = self.config.get("enabled", True)
        self.provider = self.config.get("provider", "local")
        self.model_name = self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.top_k = self.config.get("top_k", 50)
        self.batch_size = self.config.get("batch_size", 32)
        
        # Initialize provider
        if not self.enabled:
            logger.info("Reranker is disabled")
            return
        
        if self.provider == "cohere":
            self._init_cohere()
        elif self.provider == "local":
            self._init_local()
        else:
            raise RerankerConfigurationError(
                f"Invalid provider: {self.provider}. Must be 'cohere' or 'local'"
            )
    
    def _load_config(self, config: Optional[Union[Dict[str, Any], Path, str]]) -> Dict[str, Any]:
        """
        Load configuration from dict, file path, or default location.
        
        Args:
            config: Configuration dict, path, or None
        
        Returns:
            Configuration dictionary
        
        Raises:
            RerankerConfigurationError: If config file cannot be loaded
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
            raise RerankerConfigurationError(
                f"Invalid config type: {type(config)}. Expected dict, Path, str, or None"
            )
    
    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            Configuration dictionary (reranking section)
        
        Raises:
            RerankerConfigurationError: If file cannot be loaded or parsed
        """
        if not config_path.exists():
            raise RerankerConfigurationError(
                f"Config file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RerankerConfigurationError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        except Exception as e:
            raise RerankerConfigurationError(
                f"Failed to load config file {config_path}: {e}"
            ) from e
        
        if full_config is None:
            raise RerankerConfigurationError(f"Config file {config_path} is empty")
        
        # Extract reranking section
        reranking_config = full_config.get("reranking", {})
        if not reranking_config:
            logger.warning(
                f"No 'reranking' section found in {config_path}, using defaults"
            )
            return {}
        
        return reranking_config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary
        
        Raises:
            RerankerConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise RerankerConfigurationError("Config must be a dictionary")
        
        # Validate enabled
        if "enabled" in config and not isinstance(config["enabled"], bool):
            raise RerankerConfigurationError("'enabled' must be a boolean")
        
        # Validate provider
        if "provider" in config:
            if config["provider"] not in ["cohere", "local"]:
                raise RerankerConfigurationError(
                    f"Invalid provider: {config['provider']}. Must be 'cohere' or 'local'"
                )
        
        # Validate model
        if "model" in config and not isinstance(config["model"], str):
            raise RerankerConfigurationError("'model' must be a string")
        
        # Validate top_k
        if "top_k" in config:
            if not isinstance(config["top_k"], int) or config["top_k"] <= 0:
                raise RerankerConfigurationError("'top_k' must be a positive integer")
        
        # Validate batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                raise RerankerConfigurationError("'batch_size' must be a positive integer")
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from config or environment variable.
        
        Returns:
            API key string or None if not found
        """
        # Try direct API key first
        api_key = self.config.get("api_key")
        if api_key:
            return api_key
        
        # Try environment variable
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                return api_key
            else:
                logger.warning(
                    f"API key environment variable '{api_key_env}' not set"
                )
        
        return None
    
    def _init_cohere(self) -> None:
        """
        Initialize Cohere API client.
        
        Raises:
            RerankerModelError: If initialization fails
        """
        try:
            import cohere
        except ImportError:
            logger.warning("cohere package not installed, falling back to local model")
            self.provider = "local"
            self._init_local()
            return
        
        api_key = self._load_api_key()
        if not api_key:
            logger.warning("Cohere API key not found, falling back to local model")
            self.provider = "local"
            self._init_local()
            return
        
        try:
            self.client = cohere.Client(api_key=api_key)
            logger.info(f"Initialized Cohere reranker with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise RerankerModelError(f"Failed to initialize Cohere client: {e}") from e
    
    def _init_local(self) -> None:
        """
        Initialize local cross-encoder model.
        
        Raises:
            RerankerModelError: If model loading fails
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded local reranker: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers package not installed")
            raise RerankerModelError(
                "sentence-transformers package is required for local reranking"
            ) from None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise RerankerModelError(f"Failed to load reranker model: {e}") from e
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        For cross-encoder models, raw scores can be negative or unbounded.
        This method uses sigmoid normalization to map scores to [0, 1].
        
        Args:
            scores: Raw scores array
        
        Returns:
            Normalized scores in [0, 1] range
        """
        # Use sigmoid normalization: 1 / (1 + exp(-x))
        # This maps any real number to (0, 1)
        normalized = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        return normalized
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts to re-rank
            top_k: Number of top results to return (defaults to config top_k)
            batch_size: Batch size for processing (defaults to config batch_size)
        
        Returns:
            List of dicts with:
                - index: Original index of document
                - rerank_score: Relevance score in [0, 1] range
        
        Raises:
            RerankerAPIError: If API call fails (for Cohere provider)
            RerankerModelError: If model prediction fails (for local provider)
        """
        if not self.enabled:
            return [
                {"index": i, "rerank_score": 0.0}
                for i in range(len(documents))
            ]
        
        if not documents:
            return []
        
        # Handle top_k=0 explicitly (None means use default)
        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:
            return []
        
        top_k = min(top_k, len(documents))
        batch_size = batch_size or self.batch_size
        
        if self.provider == "cohere" and hasattr(self, "client"):
            return self._rerank_cohere(query, documents, top_k)
        elif self.provider == "local" and hasattr(self, "model"):
            return self._rerank_local(query, documents, top_k, batch_size)
        else:
            logger.warning("Reranker not properly initialized, returning zero scores")
            return [
                {"index": i, "rerank_score": 0.0}
                for i in range(min(top_k, len(documents)))
            ]
    
    def _rerank_cohere(
        self,
        query: str,
        documents: List[str],
        top_k: int
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Re-rank using Cohere API.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
        
        Returns:
            List of dicts with index and rerank_score
        
        Raises:
            RerankerAPIError: If API call fails
        """
        try:
            results = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            # Cohere returns scores in [0, 1] range already
            return [
                {
                    "index": r.index,
                    "rerank_score": float(r.relevance_score)
                }
                for r in results.results
            ]
        except Exception as e:
            error_msg = f"Cohere reranking failed: {e}"
            logger.error(error_msg)
            
            # Check for specific error types
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RerankerAPIError(f"Rate limit exceeded: {e}") from e
            elif "auth" in error_str or "401" in error_str or "403" in error_str:
                raise RerankerAPIError(f"Authentication failed: {e}") from e
            elif "network" in error_str or "connection" in error_str:
                raise RerankerAPIError(f"Network error: {e}") from e
            else:
                raise RerankerAPIError(error_msg) from e
    
    def _rerank_local(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        batch_size: int
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Re-rank using local cross-encoder model with batch processing.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            batch_size: Batch size for processing
        
        Returns:
            List of dicts with index and rerank_score (normalized to [0, 1])
        
        Raises:
            RerankerModelError: If model prediction fails
        """
        try:
            # Process in batches to avoid memory issues
            all_scores = []
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                pairs = [[query, doc] for doc in batch_docs]
                batch_scores = self.model.predict(pairs)
                all_scores.extend(batch_scores)
            
            # Convert to numpy array for normalization
            scores_array = np.array(all_scores)
            
            # Normalize scores to [0, 1] range
            normalized_scores = self._normalize_scores(scores_array)
            
            # Sort by score and return top_k
            scored_indices = sorted(
                enumerate(normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            return [
                {
                    "index": idx,
                    "rerank_score": float(score)
                }
                for idx, score in scored_indices
            ]
        except Exception as e:
            error_msg = f"Local reranking failed: {e}"
            logger.error(error_msg)
            raise RerankerModelError(error_msg) from e

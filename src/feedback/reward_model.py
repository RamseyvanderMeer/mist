"""
Reward model for RLHF training.

This module provides a neural network reward model that predicts feedback scores
from query and document embeddings. The model is used in Reinforcement Learning
from Human Feedback (RLHF) to score recommendation quality.

Architecture:
    768 → 512 → 256 → 1
    
    - Input: Difference of query and document embeddings (768-dim)
    - Layer 1: Linear(768, 512) + ReLU + Dropout(0.1)
    - Layer 2: Linear(512, 256) + ReLU + Dropout(0.1)
    - Layer 3: Linear(256, 1) + Sigmoid
    - Output: Reward signal in range [0.0, 1.0]

The model uses the difference between query and document embeddings to capture
their relationship, which is more efficient than concatenation and maintains
the 768-dimensional input requirement.

Example:
    >>> import torch
    >>> from src.feedback.reward_model import RewardModel
    >>> 
    >>> # Initialize model
    >>> model = RewardModel(input_dim=768, hidden_dim=512)
    >>> 
    >>> # Create sample embeddings (batch_size=2, embedding_dim=768)
    >>> query_emb = torch.randn(2, 768)
    >>> doc_emb = torch.randn(2, 768)
    >>> 
    >>> # Compute reward scores
    >>> rewards = model(query_emb, doc_emb)
    >>> print(rewards.shape)  # torch.Size([2, 1])
    >>> print(rewards)  # Values in [0.0, 1.0] range
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union

from ..paths import Paths

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Neural network reward model for RLHF.
    
    Predicts feedback scores from query and document embeddings using a
    feedforward network architecture. The model takes the difference between
    query and document embeddings as input to capture their relationship.
    
    Architecture: 768 → 512 → 256 → 1
    
    Attributes:
        input_dim: Input embedding dimension (default: 768)
        hidden_dim: Hidden layer dimension (default: 512)
        network: Sequential neural network layers
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        config: Optional[Union[Dict[str, Any], Path, str]] = None
    ) -> None:
        """
        Initialize reward model.
        
        Args:
            input_dim: Input embedding dimension. Default: 768.
            hidden_dim: Hidden layer dimension. Default: 512.
            config: Optional configuration. Can be:
                - Dict with 'reward_model' section containing 'input_dim' and 'hidden_dim'
                - Path or str to training_config.yaml file
                - None to use defaults
        
        Raises:
            ValueError: If input_dim or hidden_dim are invalid (<= 0)
        """
        super().__init__()
        
        # Load config if provided
        if config is not None:
            config_dict = self._load_config(config)
            reward_config = config_dict.get("reward_model", {})
            input_dim = reward_config.get("input_dim", input_dim)
            hidden_dim = reward_config.get("hidden_dim", hidden_dim)
        
        # Validate dimensions
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build network: 768 → 512 → 256 → 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"Initialized RewardModel with architecture: "
            f"{input_dim} → {hidden_dim} → {hidden_dim // 2} → 1"
        )
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using proper initialization schemes.
        
        Uses Xavier/Glorot uniform initialization for Linear layers, which is
        appropriate for ReLU activations and helps with training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot uniform initialization for ReLU activations
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
    
    def _load_config(
        self,
        config: Union[Dict[str, Any], Path, str]
    ) -> Dict[str, Any]:
        """
        Load configuration from dict or file path.
        
        Args:
            config: Configuration dict, Path object, or string path
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config cannot be parsed
        """
        if isinstance(config, dict):
            return config
        
        # Convert to Path if string
        if isinstance(config, str):
            config_path = Path(config)
        else:
            config_path = config
        
        # If relative path, try to resolve using Paths
        if not config_path.is_absolute():
            paths = Paths()
            # Check if it's just a filename, use training_config property
            if config_path.name == "training_config.yaml":
                config_path = paths.training_config
            else:
                config_path = paths.config / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Please ensure the file exists or provide a dict config."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to load config file {config_path}: {e}"
            ) from e
        
        if loaded_config is None:
            raise ValueError(f"Config file {config_path} is empty")
        
        return loaded_config
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        doc_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward score from query and document embeddings.
        
        Takes the difference between query and document embeddings to capture
        their relationship. This maintains the 768-dimensional input requirement
        while being more efficient than concatenation.
        
        Args:
            query_embedding: Query embedding tensor of shape (batch_size, 768)
            doc_embedding: Document embedding tensor of shape (batch_size, 768)
        
        Returns:
            Reward score tensor of shape (batch_size, 1) with values in [0.0, 1.0]
        
        Raises:
            ValueError: If embeddings have mismatched dimensions or shapes
        """
        # Validate input shapes
        if query_embedding.shape != doc_embedding.shape:
            raise ValueError(
                f"Query and document embeddings must have the same shape. "
                f"Got query: {query_embedding.shape}, doc: {doc_embedding.shape}"
            )
        
        if query_embedding.size(-1) != self.input_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.input_dim}, "
                f"got {query_embedding.size(-1)}"
            )
        
        # Use difference of embeddings to capture relationship
        # This maintains 768-dim input while being more efficient than concatenation
        combined = query_embedding - doc_embedding
        
        # Forward through network
        reward = self.network(combined)
        
        return reward

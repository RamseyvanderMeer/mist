"""
Reward model for RLHF training.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Neural network reward model for RLHF.
    """
    
    def __init__(self, input_dim=768, hidden_dim=512):
        """
        Initialize reward model.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
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
    
    def forward(self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute reward score.
        
        Args:
            query_embedding: Query embedding
            doc_embedding: Document embedding
        
        Returns:
            Reward score (0.0 to 1.0)
        """
        # Combine embeddings
        combined = torch.cat([query_embedding, doc_embedding], dim=1)
        
        # Use difference if too large
        if combined.size(1) > self.network[0].in_features:
            combined = query_embedding - doc_embedding
        
        return self.network(combined)

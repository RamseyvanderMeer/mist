"""
Contrastive learning trainer for embedding fine-tuning.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

# Import the new InfoNCELoss class
from src.learning.losses import InfoNCELoss, contrastive_loss as _legacy_contrastive_loss

logger = logging.getLogger(__name__)


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: List[torch.Tensor], temperature: float = 0.05) -> torch.Tensor:
    """
    Deprecated: Use InfoNCELoss class from src.learning.losses instead.
    
    Legacy InfoNCE loss function for backward compatibility.
    
    Args:
        anchor: Query embedding
        positive: Relevant document embedding
        negatives: List of irrelevant document embeddings
        temperature: Temperature parameter
    
    Returns:
        Contrastive loss
    
    Note:
        This function is maintained for backward compatibility. New code should
        use the InfoNCELoss class which provides better performance and features
        like hard negative mining.
    """
    # Use the legacy implementation from losses.py
    return _legacy_contrastive_loss(anchor, positive, negatives, temperature)

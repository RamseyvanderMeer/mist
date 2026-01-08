"""
Contrastive learning trainer for embedding fine-tuning.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: List[torch.Tensor], temperature: float = 0.05) -> torch.Tensor:
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        anchor: Query embedding
        positive: Relevant document embedding
        negatives: List of irrelevant document embeddings
        temperature: Temperature parameter
    
    Returns:
        Contrastive loss
    """
    # Positive pair similarity
    pos_sim = F.cosine_similarity(anchor, positive, dim=1) / temperature
    
    # Negative pair similarities
    neg_sims = []
    for neg in negatives:
        neg_sim = F.cosine_similarity(anchor, neg, dim=1) / temperature
        neg_sims.append(neg_sim)
    
    # Combine
    all_sims = torch.cat([pos_sim.unsqueeze(1)] + [n.unsqueeze(1) for n in neg_sims], dim=1)
    
    # InfoNCE loss
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(all_sims, labels)
    
    return loss

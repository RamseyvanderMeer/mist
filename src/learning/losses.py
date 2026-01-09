"""
Contrastive learning loss functions for embedding fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss for contrastive learning.
    
    This loss function is used to train embeddings by maximizing the similarity
    between positive pairs (anchor and positive) while minimizing similarity
    between negative pairs (anchor and negatives).
    
    Supports hard negative mining to select the most challenging negatives
    for improved training effectiveness.
    
    Args:
        temperature: Temperature parameter for scaling similarities. Lower values
            make the distribution sharper. Default: 0.05
        reduction: Specifies the reduction to apply to the output. 'mean' (default),
            'sum', or 'none'.
    """
    
    def __init__(self, temperature: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.temperature = temperature
        self.reduction = reduction
    
    def select_hard_negatives(
        self,
        anchor: torch.Tensor,
        negatives: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Select hard negatives by choosing the k most similar negatives to the anchor.
        
        Hard negatives are those that are most similar to the anchor but should
        still be distinguished. This makes training more effective by focusing
        on challenging examples.
        
        Args:
            anchor: Anchor embeddings of shape (batch_size, embedding_dim) or
                (embedding_dim,) for single sample
            negatives: Candidate negative embeddings of shape
                (num_candidates, embedding_dim) or (batch_size, num_candidates, embedding_dim)
            k: Number of hard negatives to select
        
        Returns:
            Selected hard negatives of shape (batch_size, k, embedding_dim) or
            (k, embedding_dim) for single sample
        """
        # Handle single sample case
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Handle negatives shape
        if negatives.dim() == 2:
            # (num_candidates, embedding_dim) -> (batch_size, num_candidates, embedding_dim)
            negatives = negatives.unsqueeze(0).expand(anchor.size(0), -1, -1)
        elif negatives.dim() == 3:
            # (batch_size, num_candidates, embedding_dim)
            pass
        else:
            raise ValueError(
                f"Negatives must be 2D or 3D tensor, got shape {negatives.shape}"
            )
        
        batch_size = anchor.size(0)
        num_candidates = negatives.size(1)
        embedding_dim = anchor.size(1)
        
        # Ensure k doesn't exceed available negatives
        k = min(k, num_candidates)
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        # Compute cosine similarities between anchor and all negatives
        # anchor: (batch_size, embedding_dim)
        # negatives: (batch_size, num_candidates, embedding_dim)
        anchor_expanded = anchor.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Compute cosine similarity: (batch_size, num_candidates)
        similarities = F.cosine_similarity(
            anchor_expanded.expand(-1, num_candidates, -1),
            negatives,
            dim=2
        )
        
        # Select top-k most similar (hardest negatives)
        # We want highest similarity, so use topk
        _, topk_indices = torch.topk(similarities, k, dim=1)  # (batch_size, k)
        
        # Gather selected negatives
        batch_indices = torch.arange(batch_size, device=anchor.device).unsqueeze(1)
        hard_negatives = negatives[batch_indices, topk_indices]  # (batch_size, k, embedding_dim)
        
        # Return to original shape if single sample
        if single_sample:
            hard_negatives = hard_negatives.squeeze(0)  # (k, embedding_dim)
        
        return hard_negatives
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        hard_negative_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor embeddings of shape (batch_size, embedding_dim) or
                (embedding_dim,) for single sample
            positive: Positive embeddings of shape (batch_size, embedding_dim) or
                (embedding_dim,) for single sample
            negatives: Negative embeddings of shape (num_negatives, embedding_dim),
                (batch_size, num_negatives, embedding_dim), or list of tensors
            hard_negative_k: If specified, select top-k hard negatives from candidates.
                If None, uses all provided negatives.
        
        Returns:
            Loss value. If reduction='mean', returns scalar. If reduction='sum',
            returns scalar sum. If reduction='none', returns tensor of shape
            (batch_size,) or scalar for single sample.
        """
        # Handle single sample case
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
            positive = positive.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Ensure all tensors are on same device
        device = anchor.device
        positive = positive.to(device)
        
        # Handle negatives input format
        if isinstance(negatives, list):
            # Convert list of tensors to single tensor
            negatives = torch.stack(negatives, dim=0)  # (num_negatives, embedding_dim)
        
        negatives = negatives.to(device)
        
        # Handle negatives shape
        if negatives.dim() == 2:
            # (num_negatives, embedding_dim) -> (batch_size, num_negatives, embedding_dim)
            negatives = negatives.unsqueeze(0).expand(anchor.size(0), -1, -1)
        elif negatives.dim() == 3:
            # (batch_size, num_negatives, embedding_dim)
            if negatives.size(0) != anchor.size(0):
                raise ValueError(
                    f"Batch size mismatch: anchor has {anchor.size(0)}, "
                    f"negatives has {negatives.size(0)}"
                )
        else:
            raise ValueError(
                f"Negatives must be 2D or 3D tensor, got shape {negatives.shape}"
            )
        
        # Validate shapes
        batch_size = anchor.size(0)
        embedding_dim = anchor.size(1)
        
        if positive.shape != anchor.shape:
            raise ValueError(
                f"Positive shape {positive.shape} must match anchor shape {anchor.shape}"
            )
        
        if negatives.size(2) != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: anchor has {embedding_dim}, "
                f"negatives has {negatives.size(2)}"
            )
        
        # Apply hard negative mining if requested
        if hard_negative_k is not None:
            if hard_negative_k <= 0:
                raise ValueError(f"hard_negative_k must be positive, got {hard_negative_k}")
            negatives = self.select_hard_negatives(anchor, negatives, hard_negative_k)
        
        num_negatives = negatives.size(1)
        
        if num_negatives == 0:
            raise ValueError("At least one negative is required")
        
        # Compute cosine similarities
        # Positive similarity: (batch_size,)
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        
        # Negative similarities: (batch_size, num_negatives)
        # Reshape anchor for broadcasting: (batch_size, 1, embedding_dim)
        anchor_expanded = anchor.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Compute similarities: (batch_size, num_negatives)
        neg_sims = F.cosine_similarity(
            anchor_expanded.expand(-1, num_negatives, -1),
            negatives,
            dim=2
        ) / self.temperature
        
        # Stack logits: positive first, then negatives
        # Shape: (batch_size, 1 + num_negatives)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        
        # Labels: positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        # Return to original shape if single sample
        if single_sample and self.reduction == 'none':
            loss = loss.squeeze(0)
        
        return loss


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: Union[torch.Tensor, list],
    temperature: float = 0.05
) -> torch.Tensor:
    """
    Deprecated: Use InfoNCELoss class instead.
    
    Legacy function for backward compatibility.
    Computes InfoNCE loss for contrastive learning.
    
    Args:
        anchor: Query embedding of shape (batch_size, embedding_dim) or (embedding_dim,)
        positive: Relevant document embedding of shape (batch_size, embedding_dim) or (embedding_dim,)
        negatives: Irrelevant document embeddings. Can be:
            - List of tensors, each of shape (embedding_dim,)
            - Tensor of shape (num_negatives, embedding_dim) or (batch_size, num_negatives, embedding_dim)
        temperature: Temperature parameter for scaling similarities
    
    Returns:
        Contrastive loss value
    """
    logger.warning(
        "contrastive_loss() is deprecated. Use InfoNCELoss class instead."
    )
    loss_fn = InfoNCELoss(temperature=temperature)
    return loss_fn(anchor, positive, negatives)

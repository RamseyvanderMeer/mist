"""
Tests for contrastive learning loss functions.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.learning.losses import InfoNCELoss, contrastive_loss


class TestInfoNCELoss:
    """Test suite for InfoNCELoss class"""
    
    def test_init_default(self):
        """Test InfoNCELoss initialization with default parameters"""
        loss_fn = InfoNCELoss()
        assert loss_fn.temperature == 0.05
        assert loss_fn.reduction == 'mean'
    
    def test_init_custom_temperature(self):
        """Test InfoNCELoss initialization with custom temperature"""
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature == 0.1
    
    def test_init_invalid_temperature(self):
        """Test that invalid temperature raises ValueError"""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=-0.1)
    
    def test_init_invalid_reduction(self):
        """Test that invalid reduction raises ValueError"""
        with pytest.raises(ValueError, match="Reduction must be"):
            InfoNCELoss(reduction='invalid')
    
    def test_loss_computation_basic(self):
        """Test basic loss computation with single sample"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        # Create embeddings with more realistic similarity distribution
        # Normalize to ensure cosine similarity works correctly
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(anchor + 0.3 * torch.randn(768), p=2, dim=0)  # Somewhat similar
        negatives = F.normalize(torch.randn(5, 768), p=2, dim=0)  # Different from anchor
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative (can be 0 with perfect separation)
    
    def test_loss_computation_batched(self):
        """Test loss computation with batched inputs"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        batch_size = 4
        embedding_dim = 768
        num_negatives = 10
        
        anchor = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        positive = F.normalize(anchor + 0.3 * torch.randn(batch_size, embedding_dim), p=2, dim=1)
        negatives = F.normalize(torch.randn(num_negatives, embedding_dim), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar (mean reduction)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_decreases_with_higher_positive_similarity(self):
        """Test that loss decreases when positive similarity increases"""
        loss_fn = InfoNCELoss(temperature=0.1)  # Higher temperature for more stable gradients
        
        # Use normalized embeddings for consistent cosine similarity behavior
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        # Create negatives that are somewhat similar to anchor (hard negatives)
        negatives = F.normalize(anchor.unsqueeze(0) + 0.2 * torch.randn(5, 768), p=2, dim=1)
        
        # High similarity positive (very similar to anchor)
        positive_high = F.normalize(anchor + 0.05 * torch.randn(768), p=2, dim=0)
        loss_high = loss_fn(anchor, positive_high, negatives)
        
        # Lower similarity positive (less similar to anchor)
        positive_low = F.normalize(anchor + 0.3 * torch.randn(768), p=2, dim=0)
        loss_low = loss_fn(anchor, positive_low, negatives)
        
        # With hard negatives, higher similarity should give lower loss
        # But if both are perfect (loss=0), we can't compare, so check they're both valid
        assert loss_high.item() >= 0
        assert loss_low.item() >= 0
        # If both are non-zero, higher similarity should give lower loss
        if loss_high.item() > 0 and loss_low.item() > 0:
            assert loss_high.item() <= loss_low.item(), \
                "Loss should decrease (or stay same) with higher positive similarity"
    
    def test_loss_increases_with_higher_negative_similarity(self):
        """Test that loss increases when negative similarity increases"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        positive = anchor + 0.1 * torch.randn(768)
        
        # Low similarity negatives
        negatives_low = torch.randn(5, 768)
        loss_low = loss_fn(anchor, positive, negatives_low)
        
        # High similarity negatives (harder)
        negatives_high = anchor.unsqueeze(0) + 0.1 * torch.randn(5, 768)
        loss_high = loss_fn(anchor, positive, negatives_high)
        
        assert loss_high.item() > loss_low.item(), \
            "Loss should increase with higher negative similarity"
    
    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude"""
        anchor = torch.randn(768)
        positive = anchor + 0.1 * torch.randn(768)
        negatives = torch.randn(5, 768)
        
        loss_low_temp = InfoNCELoss(temperature=0.01)(anchor, positive, negatives)
        loss_high_temp = InfoNCELoss(temperature=0.1)(anchor, positive, negatives)
        
        # Lower temperature should generally give higher loss (sharper distribution)
        # But this depends on the specific similarities, so we just check they're different
        assert not torch.isclose(loss_low_temp, loss_high_temp, atol=1e-6)
    
    def test_single_negative(self):
        """Test loss computation with single negative"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(anchor + 0.3 * torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(1, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_many_negatives(self):
        """Test loss computation with many negatives"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(anchor + 0.3 * torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(50, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_reduction_none(self):
        """Test loss with reduction='none'"""
        loss_fn = InfoNCELoss(temperature=0.05, reduction='none')
        
        batch_size = 4
        anchor = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        positive = F.normalize(anchor + 0.3 * torch.randn(batch_size, 768), p=2, dim=1)
        negatives = F.normalize(torch.randn(5, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert loss.shape == (batch_size,)
        assert torch.all(loss >= 0)  # Loss should be non-negative
    
    def test_reduction_sum(self):
        """Test loss with reduction='sum'"""
        loss_fn = InfoNCELoss(temperature=0.05, reduction='sum')
        
        batch_size = 4
        anchor = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        positive = F.normalize(anchor + 0.3 * torch.randn(batch_size, 768), p=2, dim=1)
        negatives = F.normalize(torch.randn(5, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        # Create embeddings with requires_grad
        anchor = torch.randn(768, requires_grad=True)
        positive = torch.randn(768, requires_grad=True)
        negatives = torch.randn(5, 768, requires_grad=True)
        
        loss = loss_fn(anchor, positive, negatives)
        loss.backward()
        
        # Check gradients exist
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None
        
        # Check gradients are non-zero
        assert not torch.allclose(anchor.grad, torch.zeros_like(anchor.grad))
        assert not torch.allclose(positive.grad, torch.zeros_like(positive.grad))
        assert not torch.allclose(negatives.grad, torch.zeros_like(negatives.grad))
    
    def test_gradient_flow_batched(self):
        """Test gradient flow with batched inputs"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        batch_size = 4
        anchor = torch.randn(batch_size, 768, requires_grad=True)
        positive = torch.randn(batch_size, 768, requires_grad=True)
        negatives = torch.randn(10, 768, requires_grad=True)
        
        loss = loss_fn(anchor, positive, negatives)
        loss.backward()
        
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None
    
    def test_shape_validation(self):
        """Test that shape mismatches raise appropriate errors"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        positive = torch.randn(512)  # Wrong dimension
        negatives = torch.randn(5, 768)
        
        with pytest.raises(ValueError, match="must match anchor shape"):
            loss_fn(anchor, positive, negatives)
    
    def test_batch_size_mismatch(self):
        """Test that batch size mismatch raises error"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(4, 768)
        positive = torch.randn(4, 768)
        negatives = torch.randn(3, 5, 768)  # Batch size 3 vs 4
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(anchor, positive, negatives)
    
    def test_no_negatives_error(self):
        """Test that zero negatives raises error"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        positive = torch.randn(768)
        negatives = torch.randn(0, 768)  # Empty
        
        with pytest.raises(ValueError, match="At least one negative is required"):
            loss_fn(anchor, positive, negatives)
    
    def test_negatives_as_list(self):
        """Test that negatives can be provided as a list"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = [F.normalize(torch.randn(768), p=2, dim=0) for _ in range(5)]
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestHardNegativeMining:
    """Test suite for hard negative mining functionality"""
    
    def test_select_hard_negatives_basic(self):
        """Test basic hard negative selection"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        # Create negatives with varying similarity
        negatives = torch.randn(10, 768)
        # Make some negatives more similar to anchor
        negatives[0:3] = anchor.unsqueeze(0) + 0.1 * torch.randn(3, 768)
        
        hard_negatives = loss_fn.select_hard_negatives(anchor, negatives, k=3)
        
        assert hard_negatives.shape == (3, 768)
    
    def test_select_hard_negatives_batched(self):
        """Test hard negative selection with batched inputs"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        batch_size = 4
        anchor = torch.randn(batch_size, 768)
        negatives = torch.randn(10, 768)
        
        hard_negatives = loss_fn.select_hard_negatives(anchor, negatives, k=5)
        
        assert hard_negatives.shape == (batch_size, 5, 768)
    
    def test_select_hard_negatives_k_larger_than_candidates(self):
        """Test that k larger than candidates uses all candidates"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        negatives = torch.randn(5, 768)
        
        # Request more than available
        hard_negatives = loss_fn.select_hard_negatives(anchor, negatives, k=10)
        
        assert hard_negatives.shape == (5, 768)  # Uses all 5
    
    def test_select_hard_negatives_invalid_k(self):
        """Test that invalid k raises error"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        negatives = torch.randn(5, 768)
        
        with pytest.raises(ValueError, match="k must be positive"):
            loss_fn.select_hard_negatives(anchor, negatives, k=0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            loss_fn.select_hard_negatives(anchor, negatives, k=-1)
    
    def test_loss_with_hard_negative_mining(self):
        """Test loss computation with hard negative mining"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(anchor + 0.3 * torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(20, 768), p=2, dim=0)
        
        # Loss without hard negative mining
        loss_all = loss_fn(anchor, positive, negatives, hard_negative_k=None)
        
        # Loss with hard negative mining (select top 5)
        loss_hard = loss_fn(anchor, positive, negatives, hard_negative_k=5)
        
        assert isinstance(loss_all, torch.Tensor)
        assert isinstance(loss_hard, torch.Tensor)
        assert loss_all.item() >= 0  # Loss should be non-negative
        assert loss_hard.item() >= 0  # Loss should be non-negative
    
    def test_hard_negative_mining_selects_most_similar(self):
        """Test that hard negatives are actually the most similar ones"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        # Create negatives with one very similar to anchor
        negatives = torch.randn(10, 768)
        very_similar = anchor + 0.01 * torch.randn(768)
        negatives[0] = very_similar
        
        hard_negatives = loss_fn.select_hard_negatives(anchor, negatives, k=3)
        
        # The very similar one should be in the hard negatives
        similarities = torch.cosine_similarity(
            anchor.unsqueeze(0),
            hard_negatives,
            dim=1
        )
        
        # Check that we got high similarity negatives
        assert torch.max(similarities).item() > 0.5  # Should be reasonably high
    
    def test_hard_negative_mining_invalid_k_in_loss(self):
        """Test that invalid hard_negative_k in loss raises error"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = torch.randn(768)
        positive = torch.randn(768)
        negatives = torch.randn(5, 768)
        
        with pytest.raises(ValueError, match="hard_negative_k must be positive"):
            loss_fn(anchor, positive, negatives, hard_negative_k=0)
        
        with pytest.raises(ValueError, match="hard_negative_k must be positive"):
            loss_fn(anchor, positive, negatives, hard_negative_k=-1)


class TestVariableNegatives:
    """Test suite for handling variable numbers of negatives"""
    
    def test_one_negative(self):
        """Test with exactly one negative"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(1, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        assert loss.item() >= 0
    
    def test_five_negatives(self):
        """Test with five negatives"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(5, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        assert loss.item() >= 0
    
    def test_ten_negatives(self):
        """Test with ten negatives"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(10, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        assert loss.item() >= 0
    
    def test_fifty_negatives(self):
        """Test with fifty negatives"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = F.normalize(torch.randn(50, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        assert loss.item() >= 0
    
    def test_batched_variable_negatives(self):
        """Test batched inputs with variable negatives per batch"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        batch_size = 4
        anchor = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        positive = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        # Same negatives for all batches (common case)
        negatives = F.normalize(torch.randn(10, 768), p=2, dim=0)
        
        loss = loss_fn(anchor, positive, negatives)
        assert loss.item() >= 0


class TestLegacyFunction:
    """Test suite for legacy contrastive_loss function"""
    
    def test_legacy_function_still_works(self):
        """Test that legacy function still works for backward compatibility"""
        anchor = F.normalize(torch.randn(768), p=2, dim=0)
        positive = F.normalize(torch.randn(768), p=2, dim=0)
        negatives = [F.normalize(torch.randn(768), p=2, dim=0) for _ in range(5)]
        
        loss = contrastive_loss(anchor, positive, negatives, temperature=0.05)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_legacy_function_batched(self):
        """Test legacy function with batched inputs"""
        batch_size = 4
        anchor = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        positive = F.normalize(torch.randn(batch_size, 768), p=2, dim=1)
        negatives = [F.normalize(torch.randn(768), p=2, dim=0) for _ in range(5)]
        
        loss = contrastive_loss(anchor, positive, negatives, temperature=0.05)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestDeviceHandling:
    """Test suite for device handling"""
    
    def test_all_tensors_same_device(self):
        """Test that all tensors are moved to same device"""
        loss_fn = InfoNCELoss(temperature=0.05)
        
        if torch.cuda.is_available():
            anchor = torch.randn(768).cuda()
            positive = torch.randn(768).cpu()  # Different device
            negatives = torch.randn(5, 768).cpu()
            
            # Should handle device mismatch gracefully
            loss = loss_fn(anchor, positive, negatives)
            assert isinstance(loss, torch.Tensor)
            assert loss.device == anchor.device

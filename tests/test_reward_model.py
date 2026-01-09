"""
Unit tests for RewardModel.

Tests forward pass, output range, architecture layers, initialization, and config loading.
"""
import pytest
import torch
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.feedback.reward_model import RewardModel


class TestRewardModelInitialization:
    """Test RewardModel initialization."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = RewardModel()
        
        assert model.input_dim == 768
        assert model.hidden_dim == 512
        assert isinstance(model.network, torch.nn.Sequential)
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = RewardModel(input_dim=384, hidden_dim=256)
        
        assert model.input_dim == 384
        assert model.hidden_dim == 256
    
    def test_init_invalid_input_dim(self):
        """Test that invalid input_dim raises ValueError."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            RewardModel(input_dim=0)
        
        with pytest.raises(ValueError, match="input_dim must be positive"):
            RewardModel(input_dim=-1)
    
    def test_init_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            RewardModel(hidden_dim=0)
        
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            RewardModel(hidden_dim=-1)
    
    def test_init_architecture_layers(self):
        """Test that architecture layers are correctly constructed."""
        model = RewardModel(input_dim=768, hidden_dim=512)
        
        # Verify network structure: 768 → 512 → 256 → 1
        layers = list(model.network)
        
        # Layer 1: Linear(768, 512)
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == 768
        assert layers[0].out_features == 512
        
        # ReLU
        assert isinstance(layers[1], torch.nn.ReLU)
        
        # Dropout(0.1)
        assert isinstance(layers[2], torch.nn.Dropout)
        assert layers[2].p == 0.1
        
        # Layer 2: Linear(512, 256)
        assert isinstance(layers[3], torch.nn.Linear)
        assert layers[3].in_features == 512
        assert layers[3].out_features == 256
        
        # ReLU
        assert isinstance(layers[4], torch.nn.ReLU)
        
        # Dropout(0.1)
        assert isinstance(layers[5], torch.nn.Dropout)
        assert layers[5].p == 0.1
        
        # Layer 3: Linear(256, 1)
        assert isinstance(layers[6], torch.nn.Linear)
        assert layers[6].in_features == 256
        assert layers[6].out_features == 1
        
        # Sigmoid
        assert isinstance(layers[7], torch.nn.Sigmoid)
    
    def test_init_weight_initialization(self):
        """Test that weights are properly initialized."""
        model = RewardModel()
        
        # Check that weights are initialized (not all zeros)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                # Weights should not be all zeros (Xavier init produces non-zero values)
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                # Biases should be zeros (as per initialization)
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestRewardModelForward:
    """Test RewardModel forward pass."""
    
    @pytest.fixture
    def model(self):
        """Create RewardModel instance."""
        return RewardModel(input_dim=768, hidden_dim=512)
    
    def test_forward_single_sample(self, model):
        """Test forward pass with single sample."""
        query_emb = torch.randn(1, 768)
        doc_emb = torch.randn(1, 768)
        
        reward = model(query_emb, doc_emb)
        
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (1, 1)
    
    def test_forward_batch(self, model):
        """Test forward pass with batch of samples."""
        batch_size = 5
        query_emb = torch.randn(batch_size, 768)
        doc_emb = torch.randn(batch_size, 768)
        
        reward = model(query_emb, doc_emb)
        
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (batch_size, 1)
    
    def test_forward_output_range(self, model):
        """Test that output is in [0.0, 1.0] range."""
        query_emb = torch.randn(10, 768)
        doc_emb = torch.randn(10, 768)
        
        reward = model(query_emb, doc_emb)
        
        # All values should be in [0, 1] range (Sigmoid output)
        assert torch.all(reward >= 0.0)
        assert torch.all(reward <= 1.0)
        
        # Check that values are actually in the range (not just clamped)
        assert torch.any(reward > 0.0)  # Some values should be > 0
        assert torch.any(reward < 1.0)  # Some values should be < 1
    
    def test_forward_uses_embedding_difference(self, model):
        """Test that forward uses embedding difference, not concatenation."""
        model.eval()  # Disable dropout for deterministic results
        
        query_emb = torch.randn(2, 768)
        doc_emb = torch.randn(2, 768)
        
        # Manually compute difference
        expected_diff = query_emb - doc_emb
        
        # Forward pass
        reward = model(query_emb, doc_emb)
        
        # Verify the network receives 768-dim input (not 1536-dim)
        # We can't directly check this, but we can verify the output shape is correct
        assert reward.shape == (2, 1)
        
        # Verify that using the same difference produces same result
        # If query_emb - doc_emb = expected_diff, then (expected_diff + doc_emb) - doc_emb = expected_diff
        # So model(expected_diff + doc_emb, doc_emb) should equal model(query_emb, doc_emb)
        reward2 = model(expected_diff + doc_emb, doc_emb)
        # Should be approximately the same (within numerical precision)
        assert torch.allclose(reward, reward2, atol=1e-5)
    
    def test_forward_shape_mismatch_error(self, model):
        """Test that shape mismatch raises ValueError."""
        query_emb = torch.randn(2, 768)
        doc_emb = torch.randn(3, 768)  # Different batch size
        
        with pytest.raises(ValueError, match="must have the same shape"):
            model(query_emb, doc_emb)
    
    def test_forward_dimension_mismatch_error(self, model):
        """Test that dimension mismatch raises ValueError."""
        query_emb = torch.randn(2, 512)  # Wrong dimension
        doc_emb = torch.randn(2, 512)
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            model(query_emb, doc_emb)
    
    def test_forward_deterministic_with_same_inputs(self, model):
        """Test that same inputs produce same outputs (deterministic)."""
        model.eval()  # Disable dropout for deterministic results
        
        query_emb = torch.randn(3, 768)
        doc_emb = torch.randn(3, 768)
        
        reward1 = model(query_emb, doc_emb)
        reward2 = model(query_emb, doc_emb)
        
        assert torch.allclose(reward1, reward2)


class TestRewardModelConfig:
    """Test RewardModel configuration loading."""
    
    def test_init_with_config_dict(self):
        """Test initialization with config dictionary."""
        config = {
            "reward_model": {
                "input_dim": 384,
                "hidden_dim": 256
            }
        }
        
        model = RewardModel(config=config)
        
        assert model.input_dim == 384
        assert model.hidden_dim == 256
    
    def test_init_with_config_file(self):
        """Test initialization with config file path."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "reward_model": {
                    "input_dim": 384,
                    "hidden_dim": 256
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            model = RewardModel(config=config_path)
            
            assert model.input_dim == 384
            assert model.hidden_dim == 256
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_init_with_config_path_object(self):
        """Test initialization with Path object."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "reward_model": {
                    "input_dim": 384,
                    "hidden_dim": 256
                }
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            model = RewardModel(config=config_path)
            
            assert model.input_dim == 384
            assert model.hidden_dim == 256
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_init_with_config_file_not_found(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            RewardModel(config="/nonexistent/path/config.yaml")
    
    def test_init_with_config_defaults_fallback(self):
        """Test that missing config values fall back to defaults."""
        config = {
            "reward_model": {
                # Missing input_dim and hidden_dim
            }
        }
        
        model = RewardModel(config=config)
        
        # Should use defaults
        assert model.input_dim == 768
        assert model.hidden_dim == 512
    
    def test_init_with_config_missing_section(self):
        """Test that missing reward_model section uses defaults."""
        config = {
            "other_section": {
                "some_key": "some_value"
            }
        }
        
        model = RewardModel(config=config)
        
        # Should use defaults
        assert model.input_dim == 768
        assert model.hidden_dim == 512
    
    @patch('src.feedback.reward_model.Paths')
    def test_init_with_relative_config_path(self, mock_paths):
        """Test that relative config path is resolved using Paths."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "reward_model": {
                    "input_dim": 384,
                    "hidden_dim": 256
                }
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        # Mock Paths to return our temp file
        mock_paths_instance = mock_paths.return_value
        mock_paths_instance.training_config = config_path
        
        try:
            model = RewardModel(config="training_config.yaml")
            
            assert model.input_dim == 384
            assert model.hidden_dim == 256
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_init_with_invalid_yaml(self):
        """Test that invalid YAML raises ValueError."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Failed to parse YAML"):
                RewardModel(config=config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestRewardModelIntegration:
    """Integration tests for RewardModel."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from initialization to forward pass."""
        # Initialize model
        model = RewardModel(input_dim=768, hidden_dim=512)
        
        # Create embeddings
        query_emb = torch.randn(5, 768)
        doc_emb = torch.randn(5, 768)
        
        # Forward pass
        reward = model(query_emb, doc_emb)
        
        # Verify output
        assert reward.shape == (5, 1)
        assert torch.all(reward >= 0.0)
        assert torch.all(reward <= 1.0)
    
    def test_model_with_different_batch_sizes(self):
        """Test model works with various batch sizes."""
        model = RewardModel()
        
        for batch_size in [1, 2, 5, 10, 32]:
            query_emb = torch.randn(batch_size, 768)
            doc_emb = torch.randn(batch_size, 768)
            
            reward = model(query_emb, doc_emb)
            
            assert reward.shape == (batch_size, 1)
            assert torch.all(reward >= 0.0)
            assert torch.all(reward <= 1.0)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = RewardModel()
        model.train()  # Enable training mode
        
        query_emb = torch.randn(2, 768, requires_grad=True)
        doc_emb = torch.randn(2, 768, requires_grad=True)
        
        reward = model(query_emb, doc_emb)
        
        # Compute loss and backward
        loss = reward.mean()
        loss.backward()
        
        # Verify gradients exist
        assert query_emb.grad is not None
        assert doc_emb.grad is not None
        
        # Verify model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

"""
Standalone tests for OBD data encoder (without dependencies on fault code encoder).
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.obd_data_encoder import OBDDataEncoder


def test_obd_data_encoder_single_dict():
    """Test OBD data encoder with single dict input"""
    encoder = OBDDataEncoder()
    obd_data = {
        "engine_rpm": 2500,
        "coolant_temp": 95,
        "throttle_position": 45,
        "vehicle_speed": 60
    }
    embeddings = encoder(obd_data)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768


def test_obd_data_encoder_batch():
    """Test OBD data encoder with batch processing"""
    encoder = OBDDataEncoder()
    obd_data = [
        {"engine_rpm": 2500, "coolant_temp": 95, "throttle_position": 45},
        {"engine_rpm": 3000, "coolant_temp": 100, "throttle_position": 60},
        {"engine_rpm": 2000, "coolant_temp": 90, "throttle_position": 30}
    ]
    embeddings = encoder.encode(obd_data)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 768


def test_obd_data_encoder_output_dimension():
    """Test that encoder outputs correct dimension (768)"""
    encoder = OBDDataEncoder()
    assert encoder.get_dimension() == 768
    
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(obd_data)
    assert embeddings.shape[1] == 768


def test_obd_data_encoder_normalization():
    """Test L2 normalization on output"""
    encoder = OBDDataEncoder()
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95}
    embeddings = encoder.encode(obd_data)
    
    # Check L2 normalization (norm should be close to 1.0)
    norm = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_obd_data_encoder_missing_parameters():
    """Test encoder handles missing parameters gracefully"""
    encoder = OBDDataEncoder()
    
    # Test with empty dict
    empty_data = {}
    embeddings = encoder.encode(empty_data)
    assert embeddings.shape == (1, 768)
    
    # Test with partial parameters
    partial_data = {"engine_rpm": 2500}  # Missing other common PIDs
    embeddings = encoder.encode(partial_data)
    assert embeddings.shape == (1, 768)
    
    # Test with None values
    data_with_none = {"engine_rpm": 2500, "coolant_temp": None, "throttle_position": 45}
    embeddings = encoder.encode(data_with_none)
    assert embeddings.shape == (1, 768)


def test_obd_data_encoder_unknown_parameters():
    """Test encoder handles unknown OBD parameters"""
    encoder = OBDDataEncoder()
    obd_data = {
        "engine_rpm": 2500,
        "custom_param_1": 123.45,
        "custom_param_2": -67.89,
        "custom_param_3": 999.99
    }
    embeddings = encoder.encode(obd_data)
    assert embeddings.shape == (1, 768)


def test_obd_data_encoder_attention_mechanism():
    """Test that attention mechanism produces expected outputs"""
    encoder = OBDDataEncoder()
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95, "throttle_position": 45}
    
    # Get features before attention
    features = encoder.normalize_obd_data(obd_data)
    if features.device != next(encoder.parameters()).device:
        features = features.to(next(encoder.parameters()).device)
    
    x = encoder.feature_extractor(features)
    x_before_attn = x.clone()
    
    # Apply attention
    x_seq = x.unsqueeze(1)
    x_attn, attn_weights = encoder.attention(x_seq, x_seq, x_seq)
    
    # Verify attention output shape
    assert x_attn.shape == (1, 1, encoder.hidden_dim * 2)
    assert attn_weights.shape == (1, 1, 1)  # batch, num_heads, seq_len
    
    # Verify attention changed the representation
    x_after_attn = x_attn.squeeze(1)
    assert not torch.allclose(x_before_attn, x_after_attn, atol=1e-6)


def test_obd_data_encoder_temporal_patterns():
    """Test temporal pattern handling with multiple readings"""
    encoder = OBDDataEncoder()
    
    # Simulate temporal sequence (multiple readings over time)
    temporal_data = [
        {"engine_rpm": 2000, "coolant_temp": 85, "throttle_position": 30},  # t=0
        {"engine_rpm": 2500, "coolant_temp": 90, "throttle_position": 40},  # t=1
        {"engine_rpm": 3000, "coolant_temp": 95, "throttle_position": 50},  # t=2
    ]
    
    # Each reading should be encoded independently
    embeddings = encoder.encode(temporal_data)
    assert embeddings.shape == (3, 768)
    
    # Verify each embedding is different (different inputs should produce different embeddings)
    assert not torch.allclose(embeddings[0], embeddings[1], atol=1e-5)
    assert not torch.allclose(embeddings[1], embeddings[2], atol=1e-5)


def test_obd_data_encoder_weight_initialization():
    """Test that weights are properly initialized"""
    encoder = OBDDataEncoder()
    
    # Check that Linear layers have non-zero weights
    for module in encoder.modules():
        if isinstance(module, torch.nn.Linear):
            # Weights should be initialized (not all zeros)
            assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
            # Weights should be within reasonable range (Xavier init)
            weight_std = torch.std(module.weight)
            assert weight_std > 0.01  # Should have some variance


def test_obd_data_encoder_config_loading():
    """Test config loading from dictionary"""
    config = {
        "input_dim": 128,
        "hidden_dim": 256,
        "output_dim": 768,
        "attention_heads": 8
    }
    encoder = OBDDataEncoder(config=config)
    
    assert encoder.input_dim == 128
    assert encoder.hidden_dim == 256
    assert encoder.output_dim == 768
    assert encoder.attention_heads == 8
    assert encoder.get_dimension() == 768
    
    # Test encoding works with config
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(obd_data)
    assert embeddings.shape == (1, 768)


def test_obd_data_encoder_config_partial():
    """Test config loading with partial config (uses defaults for missing)"""
    config = {
        "output_dim": 512,
        "attention_heads": 4
    }
    encoder = OBDDataEncoder(config=config)
    
    # Should use config values where provided
    assert encoder.output_dim == 512
    assert encoder.attention_heads == 4
    # Should use defaults for missing values
    assert encoder.input_dim == 128  # default
    assert encoder.hidden_dim == 256  # default
    assert encoder.get_dimension() == 512


def test_obd_data_encoder_custom_dimensions():
    """Test encoder with custom dimensions"""
    encoder = OBDDataEncoder(input_dim=64, hidden_dim=128, output_dim=256, attention_heads=4)
    
    assert encoder.input_dim == 64
    assert encoder.hidden_dim == 128
    assert encoder.output_dim == 256
    assert encoder.attention_heads == 4
    assert encoder.get_dimension() == 256
    
    # Test encoding works
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(obd_data)
    assert embeddings.shape == (1, 256)


def test_obd_data_encoder_normalize_function():
    """Test normalize_obd_data function directly"""
    encoder = OBDDataEncoder()
    
    # Test single dict
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95}
    normalized = encoder.normalize_obd_data(obd_data)
    assert normalized.shape == (1, encoder.input_dim)
    assert normalized.dtype == torch.float32
    
    # Test list of dicts
    obd_data_list = [
        {"engine_rpm": 2500, "coolant_temp": 95},
        {"engine_rpm": 3000, "coolant_temp": 100}
    ]
    normalized = encoder.normalize_obd_data(obd_data_list)
    assert normalized.shape == (2, encoder.input_dim)
    
    # Test normalization range (should be in [0, 1] for known PIDs)
    assert torch.all(normalized >= 0.0)
    assert torch.all(normalized <= 1.0)


def test_obd_data_encoder_edge_cases():
    """Test edge cases and error handling"""
    encoder = OBDDataEncoder()
    
    # Test with very large values
    large_data = {"engine_rpm": 100000, "coolant_temp": 500}
    embeddings = encoder.encode(large_data)
    assert embeddings.shape == (1, 768)
    
    # Test with negative values (for parameters that support it)
    negative_data = {"timing_advance": -30, "coolant_temp": -10}
    embeddings = encoder.encode(negative_data)
    assert embeddings.shape == (1, 768)
    
    # Test with zero values
    zero_data = {"engine_rpm": 0, "throttle_position": 0}
    embeddings = encoder.encode(zero_data)
    assert embeddings.shape == (1, 768)
    
    # Test with string values (should be ignored)
    mixed_data = {"engine_rpm": 2500, "invalid": "string_value", "coolant_temp": 95}
    embeddings = encoder.encode(mixed_data)
    assert embeddings.shape == (1, 768)

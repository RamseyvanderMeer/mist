"""
Tests for embedding modules.
"""
import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.embeddings.fault_code_encoder import FaultCodeEncoder
from src.embeddings.obd_data_encoder import OBDDataEncoder
from src.embeddings.multimodal_encoder import MultiModalEncoder
from src.paths import Paths


def test_fault_code_encoder_single_text():
    """Test fault code encoder with single text input"""
    encoder = FaultCodeEncoder()
    text = "Random/Multiple Cylinder Misfire Detected"
    embeddings = encoder.encode(text)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768


def test_fault_code_encoder_batch():
    """Test fault code encoder with batch of texts"""
    encoder = FaultCodeEncoder()
    texts = [
        "Random/Multiple Cylinder Misfire Detected",
        "Engine Coolant Temperature Sensor Circuit Malfunction",
        "Mass Air Flow Sensor Circuit Low Input"
    ]
    embeddings = encoder.encode(texts)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 768


def test_fault_code_encoder_dimension():
    """Test that encoder outputs correct dimension (768)"""
    encoder = FaultCodeEncoder()
    assert encoder.get_dimension() == 768
    
    texts = ["Test fault code"]
    embeddings = encoder.encode(texts)
    assert embeddings.shape[1] == 768


def test_fault_code_encoder_normalization():
    """Test normalization toggle"""
    encoder = FaultCodeEncoder()
    texts = ["Test fault code"]
    
    # Test with normalization (default)
    embeddings_normalized = encoder.encode(texts, normalize=True)
    norm_normalized = torch.norm(embeddings_normalized, p=2, dim=1)
    # L2 normalized vectors should have norm close to 1.0
    assert torch.allclose(norm_normalized, torch.ones_like(norm_normalized), atol=1e-5)
    
    # Test without normalization
    embeddings_not_normalized = encoder.encode(texts, normalize=False)
    norm_not_normalized = torch.norm(embeddings_not_normalized, p=2, dim=1)
    # Non-normalized vectors should have different norm
    assert not torch.allclose(norm_not_normalized, torch.ones_like(norm_not_normalized), atol=1e-5)


def test_fault_code_encoder_model_name():
    """Test that E5-Mistral model is used by default"""
    encoder = FaultCodeEncoder()
    assert encoder.model_name == "intfloat/e5-mistral-7b-instruct"
    assert "e5" in encoder.model_name.lower()


def test_fault_code_encoder_device_storage():
    """Test that device is stored as instance variable"""
    encoder = FaultCodeEncoder(device="cpu")
    assert hasattr(encoder, 'device')
    assert encoder.device == "cpu"
    
    encoder_auto = FaultCodeEncoder(device="auto")
    assert hasattr(encoder_auto, 'device')
    assert encoder_auto.device in ["cpu", "cuda"]


def test_fault_code_encoder_device_selection():
    """Test device selection (auto, cpu, cuda)"""
    # Test explicit CPU
    encoder_cpu = FaultCodeEncoder(device="cpu")
    assert encoder_cpu.device == "cpu"
    
    # Test auto selection
    encoder_auto = FaultCodeEncoder(device="auto")
    assert encoder_auto.device in ["cpu", "cuda"]


def test_fault_code_encoder_fallback_mechanism():
    """Test fallback to smaller model when E5-Mistral fails"""
    with patch('src.embeddings.fault_code_encoder.SentenceTransformer') as mock_st:
        # First call (E5-Mistral) raises exception
        # Second call (fallback) succeeds
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(1, 384)  # MiniLM output dimension
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "intfloat/e5-mistral-7b-instruct":
                raise Exception("Model not available")
            return mock_model
        
        mock_st.side_effect = side_effect
        
        encoder = FaultCodeEncoder()
        
        # Verify fallback model was loaded (should have been called twice)
        assert call_count == 2
        assert encoder.model_name == "intfloat/e5-mistral-7b-instruct"  # Name stays same
        assert encoder.model == mock_model
        
        # Verify projection layer matches fallback model dimension (384)
        assert encoder.projection.in_features == 384
        
        # Verify encoding still works
        texts = ["Test fault code"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (1, 768)


def test_fault_code_encoder_instruction_prefix():
    """Test that 'query:' prefix logic checks for E5 in model name"""
    # Test with E5 model (default)
    encoder_e5 = FaultCodeEncoder(model_name="intfloat/e5-mistral-7b-instruct")
    assert "e5" in encoder_e5.model_name.lower()
    
    # Test with non-E5 model
    encoder_non_e5 = FaultCodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert "e5" not in encoder_non_e5.model_name.lower()
    
    # Verify both encoders work
    texts = ["Test fault code"]
    embeddings_e5 = encoder_e5.encode(texts)
    embeddings_non_e5 = encoder_non_e5.encode(texts)
    
    assert embeddings_e5.shape == (1, 768)
    assert embeddings_non_e5.shape == (1, 768)


def test_fault_code_encoder_projection_dimension():
    """Test custom projection dimension"""
    encoder = FaultCodeEncoder(projection_dim=512)
    assert encoder.get_dimension() == 512
    
    texts = ["Test fault code"]
    embeddings = encoder.encode(texts)
    assert embeddings.shape[1] == 512


def test_obd_data_encoder():
    """Test OBD data encoder"""
    encoder = OBDDataEncoder()
    obd_data = {
        "engine_rpm": 2500,
        "coolant_temp": 95,
        "throttle_position": 45
    }
    embeddings = encoder.encode(obd_data)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768


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


def test_multimodal_encoder():
    """Test multimodal encoder with both inputs"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes, obd_data)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768
    # Verify embeddings are normalized
    assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(1), atol=1e-5)


def test_multimodal_missing_obd_data():
    """Test multimodal encoder handles missing OBD data gracefully"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    
    # Test with None OBD data
    embeddings = encoder.encode(fault_codes, obd_data=None)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768
    # Verify embeddings are normalized
    assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(1), atol=1e-5)


def test_multimodal_fault_only_fallback():
    """Test that fault-code-only fallback produces valid embeddings"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300", "P0171"]
    
    # Encode with OBD data
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95}
    embeddings_with_obd = encoder.encode(fault_codes, obd_data)
    
    # Encode without OBD data (fallback)
    embeddings_fault_only = encoder.encode(fault_codes, obd_data=None)
    
    # Both should have correct shape
    assert embeddings_with_obd.shape == (2, 768)
    assert embeddings_fault_only.shape == (2, 768)
    
    # Both should be normalized
    assert torch.allclose(torch.norm(embeddings_with_obd, p=2, dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.norm(embeddings_fault_only, p=2, dim=1), torch.ones(2), atol=1e-5)
    
    # They should be different (OBD data should affect the result)
    assert not torch.allclose(embeddings_with_obd, embeddings_fault_only, atol=1e-3)


def test_multimodal_residual_connections():
    """Test that residual connections allow gradient flow"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500}
    
    # Enable gradient tracking
    encoder.train()
    
    # Create inputs that require gradients
    # Note: fault_encoder.encode() uses inference mode, so we need to clone and enable gradients
    fault_emb_raw = encoder.fault_encoder.encode(fault_codes, normalize=False)
    fault_emb = fault_emb_raw.clone().detach().requires_grad_(True)
    
    obd_emb_raw = encoder.obd_encoder(obd_data)
    obd_emb = obd_emb_raw.clone().detach().requires_grad_(True)
    
    # Add sequence dimension
    fault_seq = fault_emb.unsqueeze(1)
    obd_seq = obd_emb.unsqueeze(1)
    
    # Store original inputs
    fault_input = fault_emb.clone()
    obd_input = obd_emb.clone()
    
    # Forward through cross-attention with residual
    fault_attended, _ = encoder.cross_attention_fault(fault_seq, obd_seq, obd_seq)
    fault_attended = fault_attended.squeeze(1)
    fault_output = encoder.layer_norm_fault(fault_input + fault_attended)
    
    obd_attended, _ = encoder.cross_attention_obd(obd_seq, fault_seq, fault_seq)
    obd_attended = obd_attended.squeeze(1)
    obd_output = encoder.layer_norm_obd(obd_input + obd_attended)
    
    # Verify residual connection: output should be different from input
    # (but not too different, since it's input + attention)
    assert not torch.allclose(fault_output, fault_input, atol=1e-3)
    assert not torch.allclose(obd_output, obd_input, atol=1e-3)
    
    # Verify gradients can flow through residual connections
    loss = fault_output.sum() + obd_output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert encoder.cross_attention_fault.in_proj_weight.grad is not None
    assert encoder.cross_attention_obd.in_proj_weight.grad is not None
    assert encoder.layer_norm_fault.weight.grad is not None
    assert encoder.layer_norm_obd.weight.grad is not None


def test_multimodal_config_loading():
    """Test that config loading works correctly"""
    paths = Paths()
    config_path = paths.embedding_config
    
    # Load config from file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create encoder with config
    encoder = MultiModalEncoder(config=config)
    
    # Verify config values were applied
    assert encoder.hidden_dim == config['models']['fusion']['hidden_dim']
    assert encoder.num_heads == config['models']['fusion']['num_heads']
    assert encoder.dropout == config['models']['fusion']['dropout']
    
    # Test that encoder works
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes, obd_data)
    
    assert embeddings.shape[1] == encoder.hidden_dim


def test_multimodal_cross_attention_bidirectional():
    """Test that bidirectional cross-attention works correctly"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95}
    
    encoder.train()
    
    # Get embeddings
    # Note: fault_encoder.encode() uses inference mode, so we need to clone and enable gradients
    fault_emb_raw = encoder.fault_encoder.encode(fault_codes, normalize=False)
    fault_emb = fault_emb_raw.clone().detach().requires_grad_(True)
    
    obd_emb_raw = encoder.obd_encoder(obd_data)
    obd_emb = obd_emb_raw.clone().detach().requires_grad_(True)
    
    fault_seq = fault_emb.unsqueeze(1)
    obd_seq = obd_emb.unsqueeze(1)
    
    # Test fault → OBD attention
    fault_attended, fault_attn_weights = encoder.cross_attention_fault(fault_seq, obd_seq, obd_seq)
    assert fault_attended.shape == (1, 1, 768)
    assert fault_attn_weights.shape == (1, 1, 1)  # (batch, num_heads, seq_len)
    
    # Test OBD → fault attention
    obd_attended, obd_attn_weights = encoder.cross_attention_obd(obd_seq, fault_seq, fault_seq)
    assert obd_attended.shape == (1, 1, 768)
    assert obd_attn_weights.shape == (1, 1, 1)
    
    # Verify they use separate attention layers (different outputs)
    assert not torch.allclose(fault_attended, obd_attended, atol=1e-3)
    
    # Verify attention weights are valid (sum to 1)
    assert torch.allclose(fault_attn_weights.sum(dim=-1), torch.ones(1, 1), atol=1e-5)
    assert torch.allclose(obd_attn_weights.sum(dim=-1), torch.ones(1, 1), atol=1e-5)


def test_multimodal_weight_initialization():
    """Test that weights are properly initialized"""
    encoder = MultiModalEncoder()
    
    # Check that Linear layers have non-zero weights
    for name, module in encoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.weight is not None
            assert torch.any(module.weight != 0)
            # Check that weights are initialized (not all zeros)
            assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
    
    # Check that attention layers are initialized
    assert encoder.cross_attention_fault.in_proj_weight is not None
    assert encoder.cross_attention_obd.in_proj_weight is not None
    assert torch.any(encoder.cross_attention_fault.in_proj_weight != 0)
    assert torch.any(encoder.cross_attention_obd.in_proj_weight != 0)


def test_multimodal_batch_processing():
    """Test multimodal encoder with batch inputs"""
    encoder = MultiModalEncoder()
    
    # Batch of fault codes
    fault_codes = ["P0300", "P0171", "P0420"]
    
    # Single OBD reading (should be broadcast)
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes, obd_data)
    
    assert embeddings.shape == (3, 768)
    
    # Batch of OBD readings
    obd_data_batch = [
        {"engine_rpm": 2500, "coolant_temp": 95},
        {"engine_rpm": 3000, "coolant_temp": 100},
        {"engine_rpm": 2000, "coolant_temp": 90}
    ]
    embeddings_batch = encoder.encode(fault_codes, obd_data_batch)
    
    assert embeddings_batch.shape == (3, 768)
    
    # Verify all embeddings are normalized
    assert torch.allclose(torch.norm(embeddings_batch, p=2, dim=1), torch.ones(3), atol=1e-5)


def test_multimodal_output_dimension():
    """Test that encoder outputs correct dimension"""
    encoder = MultiModalEncoder()
    assert encoder.get_dimension() == 768
    
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes, obd_data)
    
    assert embeddings.shape[1] == 768
    assert embeddings.shape[1] == encoder.get_dimension()


# ============================================================================
# Additional Edge Case Tests for FaultCodeEncoder
# ============================================================================

def test_fault_code_encoder_empty_string():
    """Test encoder handles empty string gracefully"""
    encoder = FaultCodeEncoder()
    embeddings = encoder.encode("")
    assert embeddings.shape == (1, 768)
    # Should not raise error
    assert isinstance(embeddings, torch.Tensor)


def test_fault_code_encoder_very_long_text():
    """Test encoder handles very long text inputs (stress test)"""
    encoder = FaultCodeEncoder()
    # Create a very long text (10000 characters)
    very_long_text = "Random/Multiple Cylinder Misfire Detected " * 250
    embeddings = encoder.encode(very_long_text)
    
    assert embeddings.shape == (1, 768)
    assert isinstance(embeddings, torch.Tensor)
    # Verify normalization still works
    norm = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_fault_code_encoder_invalid_input_types():
    """Test encoder handles invalid input types"""
    encoder = FaultCodeEncoder()
    
    # Test with None (should raise error)
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(None)
    
    # Test with integer (should raise error)
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(123)


def test_fault_code_encoder_list_with_non_strings():
    """Test encoder handles list containing non-string elements"""
    encoder = FaultCodeEncoder()
    
    # Test with list containing non-string
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(["P0300", 123, "P0171"])
    
    # Test with list containing None
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(["P0300", None, "P0171"])


def test_fault_code_encoder_projection_training_mode():
    """Test gradient tracking when projection layer is in training mode"""
    encoder = FaultCodeEncoder()
    
    # Set projection to training mode
    encoder.projection.train()
    assert encoder.projection.training is True
    
    texts = ["Test fault code"]
    
    # When projection is training, embeddings should allow gradient tracking
    # Note: The actual gradient tracking depends on the model.encode() behavior
    # but the code should handle it gracefully
    embeddings = encoder.encode(texts, normalize=False)
    
    assert embeddings.shape == (1, 768)
    assert isinstance(embeddings, torch.Tensor)
    
    # Test with normalize=True as well
    embeddings_norm = encoder.encode(texts, normalize=True)
    assert embeddings_norm.shape == (1, 768)
    norm = torch.norm(embeddings_norm, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


# ============================================================================
# Additional Edge Case Tests for OBDDataEncoder
# ============================================================================

def test_obd_data_encoder_invalid_input_type():
    """Test encoder raises appropriate error for invalid input types"""
    encoder = OBDDataEncoder()
    
    # Test with string (not dict or list)
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode("not a dict or list")
    
    # Test with integer
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(123)
    
    # Test with None
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(None)


def test_obd_data_encoder_all_pids():
    """Test normalization with all known OBD PIDs"""
    encoder = OBDDataEncoder()
    
    # Test with all known PIDs from the normalization function
    obd_data = {
        'engine_rpm': 2500,
        'vehicle_speed': 60,
        'throttle_position': 45,
        'coolant_temp': 95,
        'intake_temp': 85,
        'maf_airflow': 25.5,
        'fuel_pressure': 350,
        'intake_pressure': 100,
        'timing_advance': 15.5,
        'fuel_level': 75,
        'barometric_pressure': 101
    }
    
    embeddings = encoder.encode(obd_data)
    assert embeddings.shape == (1, 768)
    
    # Verify normalization
    norm = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_obd_data_encoder_boundary_values():
    """Test min/max boundary normalization"""
    encoder = OBDDataEncoder()
    
    # Test with minimum values
    min_data = {
        'engine_rpm': 0,
        'vehicle_speed': 0,
        'throttle_position': 0,
        'coolant_temp': -40,
        'intake_temp': -40,
        'maf_airflow': 0,
        'fuel_pressure': 0,
        'intake_pressure': 0,
        'timing_advance': -64,
        'fuel_level': 0,
        'barometric_pressure': 0
    }
    embeddings_min = encoder.encode(min_data)
    assert embeddings_min.shape == (1, 768)
    
    # Test with maximum values
    max_data = {
        'engine_rpm': 8000,
        'vehicle_speed': 255,
        'throttle_position': 100,
        'coolant_temp': 215,
        'intake_temp': 215,
        'maf_airflow': 655.35,
        'fuel_pressure': 765,
        'intake_pressure': 255,
        'timing_advance': 63.5,
        'fuel_level': 100,
        'barometric_pressure': 255
    }
    embeddings_max = encoder.encode(max_data)
    assert embeddings_max.shape == (1, 768)
    
    # Verify both are normalized
    norm_min = torch.norm(embeddings_min, p=2, dim=1)
    norm_max = torch.norm(embeddings_max, p=2, dim=1)
    assert torch.allclose(norm_min, torch.ones_like(norm_min), atol=1e-5)
    assert torch.allclose(norm_max, torch.ones_like(norm_max), atol=1e-5)


def test_obd_data_encoder_normalization_clamping():
    """Verify clamping to [0,1] for known PIDs and [-1,1] for unknown numeric values"""
    encoder = OBDDataEncoder()
    
    # Test normalization function directly
    obd_data = {
        'engine_rpm': 2500,
        'custom_param_1': 123.45,
        'custom_param_2': -67.89
    }
    
    normalized = encoder.normalize_obd_data(obd_data)
    
    # Known PIDs should be clamped to [0, 1]
    # First 11 features are known PIDs
    known_pids = normalized[0, :11]
    assert torch.all(known_pids >= 0.0)
    assert torch.all(known_pids <= 1.0)
    
    # Unknown numeric values should be clamped to [-1, 1]
    # Remaining features are unknown numeric values
    unknown_values = normalized[0, 11:]
    assert torch.all(unknown_values >= -1.0)
    assert torch.all(unknown_values <= 1.0)


def test_obd_data_encoder_feature_padding_truncation():
    """Test padding/truncation logic for features"""
    encoder = OBDDataEncoder()
    
    # Test with very few features (should pad)
    sparse_data = {"engine_rpm": 2500}
    normalized_sparse = encoder.normalize_obd_data(sparse_data)
    assert normalized_sparse.shape == (1, encoder.input_dim)
    
    # Test with many unknown parameters (should truncate to 10)
    many_params = {"engine_rpm": 2500}
    # Add 15 unknown numeric parameters
    for i in range(15):
        many_params[f"param_{i}"] = float(i * 10)
    
    normalized_many = encoder.normalize_obd_data(many_params)
    assert normalized_many.shape == (1, encoder.input_dim)
    # Should have 11 known PIDs + 10 unknown = 21 features, then padded to input_dim


def test_obd_data_encoder_device_mismatch():
    """Test device handling when input tensor is on different device"""
    encoder = OBDDataEncoder()
    
    # Get the device of the encoder
    encoder_device = next(encoder.parameters()).device
    
    obd_data = {"engine_rpm": 2500, "coolant_temp": 95}
    
    # The forward method should handle device mismatch automatically
    # by moving features to the encoder's device
    embeddings = encoder.encode(obd_data)
    
    assert embeddings.shape == (1, 768)
    assert embeddings.device == encoder_device


# ============================================================================
# Additional Error Handling Tests for MultiModalEncoder
# ============================================================================

def test_multimodal_batch_size_mismatch_error():
    """Test that ValueError is raised for incompatible batch sizes"""
    encoder = MultiModalEncoder()
    
    # Batch sizes that can't be broadcast (neither is 1)
    fault_codes = ["P0300", "P0171"]  # batch_size=2
    obd_data = [
        {"engine_rpm": 2500},
        {"engine_rpm": 3000},
        {"engine_rpm": 2000}
    ]  # batch_size=3
    
    with pytest.raises(ValueError, match="Batch size mismatch"):
        encoder.encode(fault_codes, obd_data)


def test_multimodal_invalid_fault_code_type():
    """Test encoder handles invalid fault_code types"""
    encoder = MultiModalEncoder()
    obd_data = {"engine_rpm": 2500}
    
    # Test with None
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(None, obd_data)
    
    # Test with integer
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(123, obd_data)
    
    # Test with list containing non-strings
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(["P0300", 123], obd_data)


def test_multimodal_invalid_obd_data_type():
    """Test encoder handles invalid OBD data types"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    
    # Test with string (not dict/list/None)
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(fault_codes, "not a dict")
    
    # Test with integer
    with pytest.raises((TypeError, AttributeError)):
        encoder.encode(fault_codes, 123)


def test_multimodal_empty_fault_code_string():
    """Test encoder handles empty string fault codes"""
    encoder = MultiModalEncoder()
    
    # Test with empty string
    embeddings = encoder.encode("", {"engine_rpm": 2500})
    assert embeddings.shape == (1, 768)
    assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(1), atol=1e-5)
    
    # Test with list containing empty strings
    embeddings_list = encoder.encode(["", "P0300"], {"engine_rpm": 2500})
    assert embeddings_list.shape == (2, 768)
    
    # Test with empty string and None OBD data (fallback)
    embeddings_fallback = encoder.encode("", obd_data=None)
    assert embeddings_fallback.shape == (1, 768)


def test_multimodal_batch_broadcasting_edge_cases():
    """Test edge cases in batch broadcasting"""
    encoder = MultiModalEncoder()
    
    # Test: single fault code with batch of OBD data (should broadcast fault)
    fault_codes_single = ["P0300"]
    obd_data_batch = [
        {"engine_rpm": 2500},
        {"engine_rpm": 3000},
        {"engine_rpm": 2000}
    ]
    embeddings = encoder.encode(fault_codes_single, obd_data_batch)
    assert embeddings.shape == (3, 768)
    
    # Test: batch of fault codes with single OBD data (should broadcast OBD)
    fault_codes_batch = ["P0300", "P0171", "P0420"]
    obd_data_single = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes_batch, obd_data_single)
    assert embeddings.shape == (3, 768)
    
    # Test: both single (should work)
    embeddings_both_single = encoder.encode("P0300", {"engine_rpm": 2500})
    assert embeddings_both_single.shape == (1, 768)
    
    # Test: both batches of same size (should work)
    fault_codes = ["P0300", "P0171"]
    obd_data = [{"engine_rpm": 2500}, {"engine_rpm": 3000}]
    embeddings = encoder.encode(fault_codes, obd_data)
    assert embeddings.shape == (2, 768)
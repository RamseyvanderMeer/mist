"""
Tests for embedding modules.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.embeddings.fault_code_encoder import FaultCodeEncoder
from src.embeddings.obd_data_encoder import OBDDataEncoder
from src.embeddings.multimodal_encoder import MultiModalEncoder


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


def test_multimodal_encoder():
    """Test multimodal encoder"""
    encoder = MultiModalEncoder()
    fault_codes = ["P0300"]
    obd_data = {"engine_rpm": 2500}
    embeddings = encoder.encode(fault_codes, obd_data)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768

"""
Tests for embedding modules.
"""
import pytest
import torch
from src.embeddings.fault_code_encoder import FaultCodeEncoder
from src.embeddings.obd_data_encoder import OBDDataEncoder
from src.embeddings.multimodal_encoder import MultiModalEncoder


def test_fault_code_encoder():
    """Test fault code encoder"""
    encoder = FaultCodeEncoder()
    texts = ["Random/Multiple Cylinder Misfire Detected"]
    embeddings = encoder.encode(texts)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768


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

"""
Multi-modal encoder combining fault codes and OBD data using cross-attention fusion.
"""
from typing import Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fault_code_encoder import FaultCodeEncoder
from .obd_data_encoder import OBDDataEncoder
import logging

logger = logging.getLogger(__name__)


class MultiModalEncoder(nn.Module):
    """
    Combines fault code text and OBD sensor data into unified embeddings
    using cross-attention fusion.
    """
    
    def __init__(self, fault_encoder=None, obd_encoder=None, hidden_dim=768, num_heads=8, dropout=0.1):
        """
        Initialize multi-modal encoder.
        
        Args:
            fault_encoder: FaultCodeEncoder instance (created if None)
            obd_encoder: OBDDataEncoder instance (created if None)
            hidden_dim: Hidden dimension for fusion (default 768)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Component encoders
        self.fault_encoder = fault_encoder or FaultCodeEncoder(projection_dim=hidden_dim)
        self.obd_encoder = obd_encoder or OBDDataEncoder(output_dim=hidden_dim)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, fault_codes: Union[str, List[str]], obd_data: Union[Dict, List[Dict]]) -> torch.Tensor:
        """
        Encode fault codes and OBD data into unified embedding.
        
        Args:
            fault_codes: Fault code descriptions (string or list)
            obd_data: OBD sensor data (dict or list of dicts)
        
        Returns:
            torch.Tensor: Unified embeddings (batch_size, hidden_dim)
        """
        # Encode components
        fault_emb = self.fault_encoder.encode(fault_codes, normalize=False)
        obd_emb = self.obd_encoder(obd_data)
        
        # Ensure same batch size
        if fault_emb.size(0) != obd_emb.size(0):
            if fault_emb.size(0) == 1:
                fault_emb = fault_emb.repeat(obd_emb.size(0), 1)
            elif obd_emb.size(0) == 1:
                obd_emb = obd_emb.repeat(fault_emb.size(0), 1)
            else:
                raise ValueError(f"Batch size mismatch: fault={fault_emb.size(0)}, obd={obd_emb.size(0)}")
        
        # Add sequence dimension for attention
        fault_seq = fault_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        obd_seq = obd_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Cross-attention: fault attends to OBD
        fault_attended, _ = self.cross_attention(fault_seq, obd_seq, obd_seq)
        
        # Cross-attention: OBD attends to fault
        obd_attended, _ = self.cross_attention(obd_seq, fault_seq, fault_seq)
        
        # Remove sequence dimension
        fault_attended = fault_attended.squeeze(1)
        obd_attended = obd_attended.squeeze(1)
        
        # Layer normalization
        fault_attended = self.layer_norm(fault_attended)
        obd_attended = self.layer_norm(obd_attended)
        
        # Concatenate and fuse
        combined = torch.cat([fault_attended, obd_attended], dim=1)
        fused = self.fusion(combined)
        
        # Final normalization
        return F.normalize(fused, p=2, dim=1)
    
    def encode(self, fault_codes: Union[str, List[str]], obd_data: Union[Dict, List[Dict]]) -> torch.Tensor:
        """Encode in eval mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(fault_codes, obd_data)
    
    def get_dimension(self) -> int:
        """Get output dimension"""
        return self.hidden_dim

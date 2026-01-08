"""
Multi-modal encoder combining fault codes and OBD data using cross-attention fusion.
"""
from typing import Union, List, Dict, Optional
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
    using bidirectional cross-attention fusion with residual connections.
    
    Supports graceful fallback to fault-code-only encoding when OBD data is unavailable.
    """
    
    def __init__(
        self,
        fault_encoder: Optional[FaultCodeEncoder] = None,
        obd_encoder: Optional[OBDDataEncoder] = None,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        config: Optional[Dict] = None
    ):
        """
        Initialize multi-modal encoder.
        
        Args:
            fault_encoder: FaultCodeEncoder instance (created if None)
            obd_encoder: OBDDataEncoder instance (created if None)
            hidden_dim: Hidden dimension for fusion (default 768)
            num_heads: Number of attention heads (default 8)
            dropout: Dropout rate (default 0.1)
            config: Optional dict with config values (overrides other params if provided)
                   Expected keys: models.fusion.hidden_dim, models.fusion.num_heads,
                   models.fusion.dropout, models.fault_code.*, models.obd_data.*
        """
        super().__init__()
        
        # Load config if provided
        if config is not None:
            fusion_config = config.get('models', {}).get('fusion', {})
            hidden_dim = fusion_config.get('hidden_dim', hidden_dim)
            num_heads = fusion_config.get('num_heads', num_heads)
            dropout = fusion_config.get('dropout', dropout)
        
        # Component encoders
        if fault_encoder is None:
            fault_config = config.get('models', {}).get('fault_code', {}) if config else None
            fault_encoder = FaultCodeEncoder(
                projection_dim=hidden_dim,
                device=fault_config.get('device', 'auto') if fault_config else 'auto'
            )
        
        if obd_encoder is None:
            obd_config = config.get('models', {}).get('obd_data', {}) if config else None
            obd_encoder = OBDDataEncoder(
                output_dim=hidden_dim,
                config=obd_config
            )
        
        self.fault_encoder = fault_encoder
        self.obd_encoder = obd_encoder
        
        # Bidirectional cross-attention mechanisms with separate layers
        # Fault attends to OBD
        self.cross_attention_fault = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # OBD attends to fault
        self.cross_attention_obd = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for residual connections
        self.layer_norm_fault = nn.LayerNorm(hidden_dim)
        self.layer_norm_obd = nn.LayerNorm(hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        Ensures proper weight initialization for training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for Linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
    
    def forward(
        self,
        fault_codes: Union[str, List[str]],
        obd_data: Optional[Union[Dict, List[Dict]]] = None
    ) -> torch.Tensor:
        """
        Encode fault codes and OBD data into unified embedding.
        
        Supports graceful fallback to fault-code-only encoding when OBD data is missing.
        
        Args:
            fault_codes: Fault code descriptions (string or list)
            obd_data: OBD sensor data (dict or list of dicts). If None, falls back to
                     fault-code-only encoding.
        
        Returns:
            torch.Tensor: Unified embeddings (batch_size, hidden_dim)
        """
        # Encode fault codes
        fault_emb = self.fault_encoder.encode(fault_codes, normalize=False)
        
        # Handle missing OBD data - fallback to fault-code-only encoding
        if obd_data is None:
            logger.warning("OBD data missing, falling back to fault-code-only encoding")
            # Return normalized fault embeddings
            return F.normalize(fault_emb, p=2, dim=1)
        
        # Encode OBD data
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
        
        # Store inputs for residual connections
        fault_input = fault_emb
        obd_input = obd_emb
        
        # Bidirectional cross-attention with residual connections
        # Fault attends to OBD
        fault_attended, _ = self.cross_attention_fault(fault_seq, obd_seq, obd_seq)
        fault_attended = fault_attended.squeeze(1)  # (batch, hidden_dim)
        # Residual connection + layer norm
        fault_attended = self.layer_norm_fault(fault_input + fault_attended)
        
        # OBD attends to fault
        obd_attended, _ = self.cross_attention_obd(obd_seq, fault_seq, fault_seq)
        obd_attended = obd_attended.squeeze(1)  # (batch, hidden_dim)
        # Residual connection + layer norm
        obd_attended = self.layer_norm_obd(obd_input + obd_attended)
        
        # Concatenate and fuse
        combined = torch.cat([fault_attended, obd_attended], dim=1)
        fused = self.fusion(combined)
        
        # Final normalization
        return F.normalize(fused, p=2, dim=1)
    
    def encode(
        self,
        fault_codes: Union[str, List[str]],
        obd_data: Optional[Union[Dict, List[Dict]]] = None
    ) -> torch.Tensor:
        """
        Encode in eval mode.
        
        Args:
            fault_codes: Fault code descriptions (string or list)
            obd_data: OBD sensor data (dict or list of dicts). If None, falls back to
                     fault-code-only encoding.
        
        Returns:
            torch.Tensor: Unified embeddings (batch_size, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(fault_codes, obd_data)
    
    def get_dimension(self) -> int:
        """Get output dimension"""
        return self.hidden_dim

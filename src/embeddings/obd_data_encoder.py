"""
OBD data encoder for structured sensor readings with attention mechanism.
"""
from typing import Union, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class OBDDataEncoder(nn.Module):
    """
    Encodes structured OBD data with attention to parameter relationships.
    Handles temporal patterns if multiple readings available.
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=768, attention_heads=8, config=None):
        """
        Initialize OBD data encoder.
        
        Args:
            input_dim: Number of normalized OBD features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            attention_heads: Number of attention heads
            config: Optional dict with config values (overrides other params if provided)
        """
        super().__init__()
        
        # Load config if provided
        if config is not None:
            input_dim = config.get('input_dim', input_dim)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            output_dim = config.get('output_dim', output_dim)
            attention_heads = config.get('attention_heads', attention_heads)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attention_heads = attention_heads
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Attention mechanism for parameter relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        This ensures proper weight initialization for training stability.
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
        
    def normalize_obd_data(self, obd_data: Union[Dict, List[Dict]]) -> torch.Tensor:
        """
        Normalize OBD parameters to fixed-size feature vector.
        
        Handles both single readings and temporal sequences:
        - Single dict: Single OBD reading at one time point
        - List of dicts: Multiple readings (batch processing or temporal sequence)
          When multiple readings are provided, each is normalized independently.
          For temporal pattern analysis, consider aggregating readings before encoding
          or using sequence-aware processing in downstream components.
        
        Args:
            obd_data: Dict or list of dicts with OBD parameters
                     Each dict contains OBD parameter names as keys and values
        
        Returns:
            torch.Tensor: Normalized feature tensor (batch_size, input_dim)
        """
        if isinstance(obd_data, dict):
            obd_data = [obd_data]
        
        # Common OBD PIDs with known ranges
        pid_ranges = {
            'engine_rpm': (0, 8000),
            'vehicle_speed': (0, 255),
            'throttle_position': (0, 100),
            'coolant_temp': (-40, 215),
            'intake_temp': (-40, 215),
            'maf_airflow': (0, 655.35),
            'fuel_pressure': (0, 765),
            'intake_pressure': (0, 255),
            'timing_advance': (-64, 63.5),
            'fuel_level': (0, 100),
            'barometric_pressure': (0, 255),
        }
        
        normalized_features = []
        for data in obd_data:
            features = []
            
            # Normalize known PIDs
            for pid, (min_val, max_val) in pid_ranges.items():
                value = data.get(pid, 0.0)
                if value is None:
                    value = 0.0
                # Normalize to [0, 1]
                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
                features.append(max(0.0, min(1.0, normalized)))  # Clamp to [0, 1]
            
            # Handle additional numeric values (limit to 10 features)
            numeric_values = [v for k, v in data.items() 
                            if k not in pid_ranges and isinstance(v, (int, float)) and v is not None]
            for val in numeric_values[:10]:
                # Normalize by max absolute value
                normalized = val / max(abs(val), 1.0) if val != 0 else 0.0
                features.append(max(-1.0, min(1.0, normalized)))
            
            # Pad or truncate to input_dim
            while len(features) < self.feature_extractor[0].in_features:
                features.append(0.0)
            features = features[:self.feature_extractor[0].in_features]
            
            normalized_features.append(features)
        
        return torch.tensor(normalized_features, dtype=torch.float32)
    
    def forward(self, obd_data: Union[Dict, List[Dict]]) -> torch.Tensor:
        """
        Encode OBD data into embeddings.
        
        Processes single readings or batches of readings. When multiple readings
        are provided (as a list of dicts), each reading is encoded independently.
        The attention mechanism captures relationships between OBD parameters
        within each reading.
        
        For temporal sequences (multiple readings over time), each time step
        is encoded separately. To capture temporal patterns, consider:
        - Pre-aggregating readings (mean, max, etc.) before encoding
        - Using sequence-aware models downstream
        - Processing temporal sequences with specialized temporal encoders
        
        Args:
            obd_data: Dict or list of dicts with OBD parameters
                     - Single dict: One OBD reading
                     - List of dicts: Batch of readings or temporal sequence
        
        Returns:
            torch.Tensor: (batch_size, output_dim) embeddings with L2 normalization
        """
        # Normalize OBD parameters
        features = self.normalize_obd_data(obd_data)
        
        if features.device != next(self.parameters()).device:
            features = features.to(next(self.parameters()).device)
        
        # Feature extraction
        x = self.feature_extractor(features)
        
        # Self-attention for parameter relationships
        x = x.unsqueeze(1)  # Add sequence dimension (batch, seq_len=1, hidden_dim)
        x_attn, _ = self.attention(x, x, x)
        x = x_attn.squeeze(1)
        
        # Output projection
        x = self.output_proj(x)
        
        return F.normalize(x, p=2, dim=1)
    
    def encode(self, obd_data: Union[Dict, List[Dict]]) -> torch.Tensor:
        """Encode in eval mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(obd_data)
    
    def get_dimension(self) -> int:
        """Get output dimension"""
        return self.output_proj.out_features

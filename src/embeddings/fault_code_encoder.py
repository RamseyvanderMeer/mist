"""
Fault code encoder using E5-Mistral-7B-Instruct for superior semantic understanding.
"""
from typing import Union, List
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class FaultCodeEncoder:
    """
    Encodes fault code descriptions using E5-Mistral-7B-Instruct.
    Provides superior semantic understanding compared to standard sentence-transformers.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct", device: str = "auto", projection_dim: int = 768) -> None:
        """
        Initialize fault code encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cpu", "cuda")
            projection_dim: Output dimension (default 768)
        """
        self.model_name = model_name
        self.projection_dim = projection_dim
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Store device as instance variable
        self.device = device
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            # E5-Mistral outputs 4096-dim, need projection layer
            self.projection = torch.nn.Linear(4096, projection_dim).to(device)
            logger.info(f"Loaded fault code encoder: {model_name} on {device}")
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA out of memory loading {model_name}. Falling back to CPU: {e}")
            # Clear cache and switch to CPU
            if device == "cuda":
                torch.cuda.empty_cache()
            device = "cpu"
            self.device = device
            try:
                self.model = SentenceTransformer(model_name, device=device)
                self.projection = torch.nn.Linear(4096, projection_dim).to(device)
                logger.info(f"Loaded fault code encoder: {model_name} on CPU (after CUDA OOM)")
            except Exception as e2:
                logger.warning(f"Failed to load {model_name} on CPU, falling back to all-MiniLM-L6-v2: {e2}")
                # Fallback to smaller model
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
                self.projection = torch.nn.Linear(384, projection_dim).to(device)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2: {e}")
            # Fallback to smaller model
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            self.projection = torch.nn.Linear(384, projection_dim).to(device)
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Encode fault code descriptions.
        
        Args:
            texts: Single string or list of strings
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            torch.Tensor: (batch_size, projection_dim) embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # E5-Mistral requires instruction prefix
        if "e5" in self.model_name.lower():
            prefixed_texts = [f"query: {text}" for text in texts]
        else:
            prefixed_texts = texts
        
        # Encode
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=True)
        
        # Clone embeddings to allow gradient tracking if projection layer is in train mode
        # This is necessary because sentence-transformers uses inference mode
        if self.projection.training:
            embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Project to target dimension
        embeddings = self.projection(embeddings)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get output dimension"""
        return self.projection_dim

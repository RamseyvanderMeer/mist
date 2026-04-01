"""
Qwen3 Embedding API encoder for repair guides.
Optimized for automotive repair guide retrieval.
Uses Nebius API (OpenAI-compatible).
"""
from typing import Union, List
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Qwen3Encoder:
    """
    Encodes text using Qwen3-Embedding-8B via Nebius API.
    
    Cost: $0.01 per million tokens (very cheap!)
    Output: 4096 dimensions (or can be reduced)
    Provider: Nebius (OpenAI-compatible API)
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = "Qwen/Qwen3-Embedding-8B",
        output_dim: int = 4096,  # Can reduce to 1024 or 768
        normalize: bool = True
    ):
        """
        Initialize Qwen3 encoder.
        
        Args:
            api_key: API key for Nebius service
            api_base: API endpoint base URL
            model: Model name (default: Qwen/Qwen3-Embedding-8B)
            output_dim: Output dimensions (4096, 1024, or 768)
            normalize: Whether to L2 normalize embeddings
        """
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        self.api_base = api_base or os.environ.get("NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/")
        self.model = model
        self.output_dim = output_dim
        self.normalize = normalize
        
        if not self.api_key:
            raise ValueError("NEBIUS_API_KEY not set")
        
        # Import OpenAI client here to avoid dependency issues
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError("openai package required. Run: pip install openai")
        
        logger.info(f"Using Qwen3 Embedding API via Nebius: {model}")
        logger.info(f"Output dimensions: {output_dim}")
        logger.info(f"Cost: ~$0.01 per million tokens")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = None,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Override normalize setting
            is_query: Whether these are queries (for instruction formatting)
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Qwen3 supports instruction-based retrieval
        # For queries, add instruction prefix
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        # Use OpenAI client
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.output_dim
        )
        
        embeddings = np.array([item.embedding for item in response.data])
        
        # Normalize if requested
        should_normalize = normalize if normalize is not None else self.normalize
        if should_normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def eval(self):
        """No-op for API compatibility."""
        return self


# Test function
if __name__ == "__main__":
    encoder = Qwen3Encoder()
    
    texts = [
        "Engine misfire in cylinder 1",
        "Oxygen sensor bank 1 sensor 2 slow response"
    ]
    
    embeddings = encoder.encode(texts, is_query=True)
    print(f"Encoded {len(texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample: {embeddings[0][:5]}")

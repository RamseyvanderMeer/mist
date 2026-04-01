"""
OpenRouter Embedding Encoder for MIST

This module provides API-based embeddings using OpenRouter,
supporting text-embedding-3-small and other models.
"""

import os
import logging
from typing import Union, List
import numpy as np
import requests

logger = logging.getLogger(__name__)


class OpenRouterEncoder:
    """
    Encoder using OpenRouter API for embeddings.
    
    Supports OpenAI embedding models via OpenRouter:
    - text-embedding-3-small (1536 dims, cheapest)
    - text-embedding-3-large (3072 dims, best quality)
    - text-embedding-ada-002 (1536 dims, legacy)
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/text-embedding-3-small",
        batch_size: int = 100
    ):
        """
        Initialize OpenRouter encoder.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model identifier (default: openai/text-embedding-3-small)
            batch_size: Max texts per API call
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")
        
        self.model = model
        self.batch_size = batch_size
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        # Determine dimensions based on model
        self.dimensions = self._get_dimensions(model)
        
        logger.info(f"Initialized OpenRouterEncoder: model={model}, dims={self.dimensions}")
    
    def _get_dimensions(self, model: str) -> int:
        """Get embedding dimensions for model."""
        if "3-large" in model:
            return 3072
        elif "3-small" in model or "ada-002" in model:
            return 1536
        else:
            return 1536  # Default
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Numpy array of embeddings (2D if list input, 1D if single)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings[0] if return_single else embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts via OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            # Log cost if available
            usage = data.get("usage", {})
            cost = usage.get("cost")
            if cost:
                logger.debug(f"Embedding cost: ${float(cost):.6f}")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {e}")
            raise
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    @property
    def output_dim(self) -> int:
        """Return output embedding dimension."""
        return self.dimensions


# Convenience function
def create_encoder(model: str = "openai/text-embedding-3-small") -> OpenRouterEncoder:
    """Create OpenRouter encoder with specified model."""
    return OpenRouterEncoder(model=model)


if __name__ == "__main__":
    # Test the encoder
    import os
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    encoder = OpenRouterEncoder(api_key=api_key)
    
    # Test single text
    text = "P0301 cylinder 1 misfire detected"
    emb = encoder.encode(text)
    print(f"Single text embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    
    # Test batch
    texts = [
        "P0171 system too lean bank 1",
        "P0420 catalytic converter efficiency below threshold",
        "P0300 random multiple cylinder misfire"
    ]
    embs = encoder.encode(texts)
    print(f"Batch embeddings: shape={embs.shape}")
    print(f"Average norm: {np.mean([np.linalg.norm(e) for e in embs]):.4f}")

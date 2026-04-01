"""
SambaNova Inference API encoder for repair guides.
Uses SambaNova's hosted API - extremely fast and cheap!
$0.13 per million tokens vs HF's ~$1+ per million.
"""
from typing import Union, List
import os
import requests
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SambaNovaEncoder:
    """
    Encodes text using SambaNova Inference API.
    Falls back to local sentence-transformers if API fails.
    
    Pricing: $0.13 per million tokens (as of 2025)
    Much cheaper than OpenAI, HuggingFace, or other providers!
    """
    
    # SambaNova API endpoint
    API_BASE = "https://api.sambanova.ai/v1"
    
    # Available embedding models on SambaNova
    # Note: SambaNova uses model IDs directly, not HuggingFace paths
    AVAILABLE_MODELS = {
        "e5-mistral-7b-instruct": "E5-Mistral-7B-Instruct",
        "E5-Mistral-7B-Instruct": "E5-Mistral-7B-Instruct",
    }
    
    def __init__(
        self,
        model_name: str = "e5-mistral-7b-instruct",
        projection_dim: int = 768,
        api_key: str = None,
        use_api: bool = True
    ):
        """
        Initialize SambaNova encoder.
        
        Args:
            model_name: Model name (see AVAILABLE_MODELS)
            projection_dim: Output dimension (default 768)
            api_key: SambaNova API key (or from env SAMBANOVA_API_KEY)
            use_api: Whether to use SambaNova API (True) or local model (False)
        """
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_api = use_api
        self.api_key = api_key or os.environ.get("SAMBANOVA_API_KEY")
        
        if self.use_api and not self.api_key:
            raise ValueError("SAMBANOVA_API_KEY not set. API-only mode requires SAMBANOVA_API_KEY environment variable.")
        
        logger.info(f"Using SambaNova Inference API: {model_name}")
        logger.info(f"Pricing: ~$0.13 per million tokens")
    
    def eval(self):
        """No-op for API compatibility with PyTorch models."""
        return self
    
    def _get_local_encoder(self):
        """Lazy load local encoder as fallback."""
        if self._local_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a smaller model for local fallback
                model_name = "all-MiniLM-L6-v2"  # 22MB, fast, good quality
                logger.info(f"Loading local model: {model_name}")
                self._local_encoder = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        return self._local_encoder
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            is_query: Whether these are queries (for instruction formatting)
        
        Returns:
            numpy array of embeddings (shape: [n_texts, projection_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Format for E5-Mistral instruction tuning
        if is_query:
            texts = [f"Instruct: Given a search query, retrieve relevant passages\nQuery: {t}" for t in texts]
        else:
            texts = [f"Passage: {t}" for t in texts]
        
        return self._encode_via_api(texts, normalize)
    
    def _encode_via_api(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using SambaNova Inference API."""
        # SambaNova uses OpenAI-compatible API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Map model name to SambaNova format
        model = self.AVAILABLE_MODELS.get(self.model_name, self.model_name)
        
        # SambaNova embedding endpoint
        response = requests.post(
            f"{self.API_BASE}/embeddings",
            headers=headers,
            json={
                "model": model,
                "input": texts,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract embeddings from response
        # Format: {"data": [{"embedding": [...], "index": 0}, ...]}
        embeddings_list = [item["embedding"] for item in result["data"]]
        embeddings = np.array(embeddings_list)
        
        # Project to desired dimension if needed
        if embeddings.shape[1] != self.projection_dim:
            factor = embeddings.shape[1] // self.projection_dim
            if factor > 0:
                embeddings = embeddings[:, :self.projection_dim * factor].reshape(
                    embeddings.shape[0], self.projection_dim, factor
                ).mean(axis=2)
            else:
                # Pad if needed
                padding = np.zeros((embeddings.shape[0], self.projection_dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _encode_local(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using local sentence-transformers model."""
        encoder = self._get_local_encoder()
        embeddings = encoder.encode(texts, normalize_embeddings=normalize)
        
        # Project to desired dimension if needed
        if embeddings.shape[1] != self.projection_dim:
            factor = embeddings.shape[1] // self.projection_dim
            if factor > 0:
                embeddings = embeddings[:, :self.projection_dim * factor].reshape(
                    embeddings.shape[0], self.projection_dim, factor
                ).mean(axis=2)
            else:
                padding = np.zeros((embeddings.shape[0], self.projection_dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
        
        return embeddings


# Convenience function for quick testing
def test_encoder():
    """Test the SambaNova encoder."""
    encoder = SambaNovaEncoder()
    
    texts = [
        "Engine misfire detected in cylinder 1",
        "Oxygen sensor bank 1 sensor 2 slow response"
    ]
    
    embeddings = encoder.encode(texts, normalize=True)
    print(f"Encoded {len(texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test similarity
    query_emb = encoder.encode(["engine misfire"], normalize=True, is_query=True)
    similarities = np.dot(embeddings, query_emb.T).flatten()
    print(f"\nSimilarities to 'engine misfire': {similarities}")


if __name__ == "__main__":
    test_encoder()

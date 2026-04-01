"""
Hugging Face Inference API encoder for repair guides.
Uses HF's hosted API instead of local model loading - no GPU required!
"""
from typing import Union, List
import os
import requests
import logging
import numpy as np

logger = logging.getLogger(__name__)


class HuggingFaceEncoder:
    """
    Encodes text using Hugging Face Inference API.
    Falls back to local sentence-transformers if API fails.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        projection_dim: int = 768,
        api_token: str = None,
        use_api: bool = True
    ):
        """
        Initialize HF encoder.
        
        Args:
            model_name: HuggingFace model name
            projection_dim: Output dimension (default 768)
            api_token: HuggingFace API token (or from env HUGGINGFACE_API_TOKEN)
            use_api: Whether to use HF API (True) or local model (False)
        """
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_api = use_api
        self.api_token = api_token or os.environ.get("HUGGINGFACE_API_TOKEN")
        
        if self.use_api and not self.api_token:
            logger.warning("HUGGINGFACE_API_TOKEN not set, will fall back to local model")
            self.use_api = False
        
        self._local_encoder = None
        
        if self.use_api:
            logger.info(f"Using HuggingFace Inference API: {model_name}")
        else:
            logger.info("Will use local sentence-transformers model")
    
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
        
        if self.use_api:
            try:
                return self._encode_via_api(texts, normalize)
            except Exception as e:
                logger.warning(f"HF API failed: {e}, falling back to local model")
                self.use_api = False
        
        return self._encode_local(texts, normalize)
    
    def _encode_via_api(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using HuggingFace Inference API."""
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # HF API expects a list of strings
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        
        # Response is a list of embeddings
        embeddings = np.array(response.json())
        
        # Project to desired dimension if needed
        if embeddings.shape[1] != self.projection_dim:
            # Simple mean pooling to reduce dimensions
            # For 4096 -> 768, we average every 5.33 dimensions
            factor = embeddings.shape[1] // self.projection_dim
            embeddings = embeddings[:, :self.projection_dim * factor].reshape(
                embeddings.shape[0], self.projection_dim, factor
            ).mean(axis=2)
        
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
            embeddings = embeddings[:, :self.projection_dim * factor].reshape(
                embeddings.shape[0], self.projection_dim, factor
            ).mean(axis=2)
        
        return embeddings


# Convenience function for quick testing
def test_encoder():
    """Test the HF encoder."""
    encoder = HuggingFaceEncoder()
    
    texts = [
        "Engine misfire detected in cylinder 1",
        "Oxygen sensor bank 1 sensor 2 slow response"
    ]
    
    embeddings = encoder.encode(texts, normalize=True)
    print(f"Encoded {len(texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")


if __name__ == "__main__":
    test_encoder()

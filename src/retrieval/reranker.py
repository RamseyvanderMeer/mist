"""
Re-ranking module using Cohere API or local cross-encoder models.
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ranks retrieval results using cross-encoder or Cohere API.
    """
    
    def __init__(self, config: dict):
        """
        Initialize reranker.
        
        Args:
            config: Configuration dict with:
                - enabled: bool
                - provider: "cohere" or "local"
                - model: Model name
                - top_k: Number of results to re-rank
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.provider = config.get("provider", "local")
        self.model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        
        if not self.enabled:
            return
        
        if self.provider == "cohere":
            try:
                import cohere
                api_key = config.get("api_key") or None
                self.client = cohere.Client(api_key=api_key) if api_key else None
                if not self.client:
                    logger.warning("Cohere API key not found, falling back to local model")
                    self.provider = "local"
            except ImportError:
                logger.warning("cohere package not installed, falling back to local model")
                self.provider = "local"
        
        if self.provider == "local":
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Loaded local reranker: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                self.enabled = False
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[dict]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
        
        Returns:
            List of dicts with rerank_score and original index
        """
        if not self.enabled or not documents:
            return [{"index": i, "rerank_score": 0.0} for i in range(len(documents))]
        
        top_k = top_k or self.config.get("top_k", 10)
        
        if self.provider == "cohere" and self.client:
            try:
                results = self.client.rerank(
                    model=self.model_name,
                    query=query,
                    documents=documents,
                    top_n=top_k
                )
                return [
                    {
                        "index": r.index,
                        "rerank_score": r.relevance_score
                    }
                    for r in results.results
                ]
            except Exception as e:
                logger.error(f"Cohere reranking failed: {e}")
                return [{"index": i, "rerank_score": 0.0} for i in range(min(top_k, len(documents)))]
        
        elif self.provider == "local" and hasattr(self, "model"):
            try:
                # Create query-document pairs
                pairs = [[query, doc] for doc in documents]
                scores = self.model.predict(pairs)
                
                # Sort by score and return top_k
                scored_indices = sorted(
                    enumerate(scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                
                return [
                    {
                        "index": idx,
                        "rerank_score": float(score)
                    }
                    for idx, score in scored_indices
                ]
            except Exception as e:
                logger.error(f"Local reranking failed: {e}")
                return [{"index": i, "rerank_score": 0.0} for i in range(min(top_k, len(documents)))]
        
        return [{"index": i, "rerank_score": 0.0} for i in range(min(top_k, len(documents)))]

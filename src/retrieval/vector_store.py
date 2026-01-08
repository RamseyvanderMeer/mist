"""
Vector store interface using Qdrant for repair guide embeddings.
"""
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant-based vector store for repair guide embeddings.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize vector store.
        
        Args:
            config: Configuration dict with:
                - provider: "qdrant"
                - collection_name: Collection name
                - distance_metric: "cosine"
                - vector_size: Vector dimension
                - url: Qdrant URL (or path for local)
        """
        self.config = config
        self.collection_name = config.get("collection_name", "repair_guides")
        
        # Initialize Qdrant client
        url = config.get("url", "http://localhost:6333")
        if url.startswith("http"):
            self.client = QdrantClient(url=url)
        else:
            # Local file-based storage
            self.client = QdrantClient(path=url)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.get("vector_size", 768),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict], batch_size: int = 100):
        """
        Add documents to vector store.
        
        Args:
            embeddings: numpy array of embeddings (n_docs, vector_size)
            documents: List of document dicts with metadata
        """
        points = []
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            point = PointStruct(
                id=doc.get("id", i),
                vector=embedding.tolist(),
                payload={
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "procedure_id": doc.get("procedure_id", ""),
                    "procedure_name": doc.get("procedure_name", ""),
                    "fault_codes": doc.get("fault_codes", []),
                    "ecu_category": doc.get("ecu_category", ""),
                    "metadata": doc.get("metadata", {})
                }
            )
            points.append(point)
        
        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Added {len(points)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of result dicts with score and metadata
        """
        # Convert filter_dict to Qdrant filter format if needed
        query_filter = None
        if filter_dict:
            # Simple exact match filter (can be extended)
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": str(result.id),
                "score": result.score,
                **result.payload
            })
        
        return formatted_results
    
    def update_document(self, doc_id: str, embedding: Optional[np.ndarray] = None, payload: Optional[Dict] = None):
        """Update document embedding or metadata"""
        update_dict = {}
        if embedding is not None:
            update_dict["vector"] = embedding.tolist()
        if payload is not None:
            update_dict["payload"] = payload
        
        if update_dict:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload or {},
                points=[doc_id]
            )
            if embedding is not None:
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=doc_id, vector=embedding.tolist())]
                )
    
    def get_collection_info(self) -> Dict:
        """Get collection information"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "config": info.config.dict()
        }

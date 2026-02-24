"""
Vector store interface using Qdrant for repair guide embeddings.
"""
from typing import List, Dict, Optional, Union, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
import numpy as np
import logging
import uuid
import os

logger = logging.getLogger(__name__)


# Exception hierarchy for vector store errors
class VectorStoreError(Exception):
    """Base exception for all vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised for connection-related errors."""
    pass


class VectorStoreConfigurationError(VectorStoreError):
    """Exception raised for configuration-related errors."""
    pass


class VectorStoreOperationError(VectorStoreError):
    """Exception raised for operation errors (add, update, delete, search)."""
    pass


class VectorStore:
    """
    Qdrant-based vector store for repair guide embeddings.
    
    This class wraps the Qdrant client with a MIST-specific interface for storing
    and retrieving repair guide embeddings. The QdrantClient handles connection
    pooling internally and is thread-safe, so a single instance can be reused
    across multiple operations.
    
    Attributes:
        config: Configuration dictionary
        collection_name: Name of the Qdrant collection
        client: QdrantClient instance (handles connection pooling internally)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize vector store.
        
        Args:
            config: Configuration dict with:
                - provider: "qdrant"
                - collection_name: Collection name
                - distance_metric: "cosine", "euclidean", or "dot"
                - vector_size: Vector dimension (default: 768)
                - url: Qdrant URL (or path for local)
                - api_key: Optional API key for cloud Qdrant (can also use QDRANT_API_KEY env var)
        
        Environment Variables:
            QDRANT_URL: Overrides config URL for cloud deployments
            QDRANT_API_KEY: API key for cloud Qdrant (if required)
        
        Raises:
            VectorStoreConfigurationError: If configuration is invalid
            VectorStoreConnectionError: If connection to Qdrant fails
        """
        self.config = config
        self.collection_name = config.get("collection_name", "repair_guides")
        
        # Validate configuration
        if not isinstance(self.collection_name, str) or not self.collection_name:
            raise VectorStoreConfigurationError(
                "collection_name must be a non-empty string"
            )
        
        vector_size = config.get("vector_size", 768)
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise VectorStoreConfigurationError(
                f"vector_size must be a positive integer, got: {vector_size}"
            )
        
        # Initialize Qdrant client
        # Support environment variable override for cloud deployments
        env_url = os.getenv("QDRANT_URL")
        env_api_key = os.getenv("QDRANT_API_KEY")
        url = env_url or config.get("url", "http://localhost:6333")
        api_key = env_api_key or config.get("api_key")
        
        # Log which configuration source is being used
        if env_url:
            logger.info(f"Using QDRANT_URL from environment: {env_url}")
            if env_api_key:
                logger.info("Using QDRANT_API_KEY from environment")
            else:
                logger.info("No QDRANT_API_KEY in environment, using config or none")
        else:
            logger.info(f"Using Qdrant URL from config: {url}")
            if api_key:
                logger.info("Using QDRANT_API_KEY from config")
        
        try:
            if url.startswith("http"):
                # Cloud or remote Qdrant server
                logger.info(f"Connecting to cloud Qdrant at: {url}")
                if api_key:
                    self.client = QdrantClient(url=url, api_key=api_key)
                else:
                    self.client = QdrantClient(url=url)
            else:
                # Local file-based storage
                logger.info(f"Using local Qdrant storage at: {url}")
                self.client = QdrantClient(path=url)
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {url}: {e}")
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant: {e}"
            ) from e
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _get_distance_metric(self) -> Distance:
        """
        Get distance metric from config.
        
        Returns:
            Distance enum value
            
        Raises:
            VectorStoreConfigurationError: If distance_metric is invalid
        """
        metric = self.config.get("distance_metric", "cosine").lower()
        metric_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        
        if metric not in metric_map:
            raise VectorStoreConfigurationError(
                f"Invalid distance_metric: {metric}. Must be one of: {list(metric_map.keys())}"
            )
        
        return metric_map[metric]
    
    def _ensure_collection(self) -> None:
        """
        Create collection if it doesn't exist.
        
        Raises:
            VectorStoreConnectionError: If connection fails
            VectorStoreConfigurationError: If collection creation fails
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                vector_size = self.config.get("vector_size", 768)
                distance = self._get_distance_metric()
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                logger.info(
                    f"Created collection: {self.collection_name} "
                    f"(size={vector_size}, distance={distance})"
                )
        except VectorStoreConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                raise VectorStoreConnectionError(
                    f"Failed to connect to Qdrant: {e}"
                ) from e
            else:
                raise VectorStoreConfigurationError(
                    f"Failed to create collection: {e}"
                ) from e
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to vector store.
        
        Args:
            embeddings: numpy array of embeddings (n_docs, vector_size)
            documents: List of document dicts with metadata. Each dict should contain:
                - id: Document ID (optional, defaults to index)
                - text: Document text
                - title: Document title (optional)
                - procedure_id: Procedure ID (optional)
                - procedure_name: Procedure name (optional)
                - fault_codes: List of fault codes (optional)
                - ecu_category: ECU category (optional)
                - metadata: Additional metadata dict (optional)
            batch_size: Number of documents to add per batch
        
        Raises:
            VectorStoreOperationError: If add operation fails
        """
        if len(embeddings) != len(documents):
            raise VectorStoreOperationError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of documents ({len(documents)})"
            )
        
        try:
            points = []
            for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
                # Convert ID to UUID if it's not already a UUID
                doc_id = doc.get("id", i)
                if isinstance(doc_id, (int, str)) and not isinstance(doc_id, uuid.UUID):
                    # Use uuid5 to create deterministic UUID from procedure_id or index
                    # Using a fixed namespace UUID for consistency
                    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                    doc_id = uuid.uuid5(namespace, str(doc_id))
                
                point = PointStruct(
                    id=doc_id,
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
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise VectorStoreOperationError(f"Failed to add documents: {e}") from e
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to vector store.
        
        .. deprecated:: Use add() instead.
        
        Args:
            embeddings: numpy array of embeddings (n_docs, vector_size)
            documents: List of document dicts with metadata
            batch_size: Number of documents to add per batch
        """
        import warnings
        warnings.warn(
            "add_documents() is deprecated. Use add() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.add(embeddings, documents, batch_size)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            top_k: Number of results to return
            filter_dict: Optional metadata filters. Supports:
                - Exact match: {"key": "value"}
                - List matching: {"fault_codes": ["code1", "code2"]}
                - Multiple conditions: All conditions must match (AND logic)
        
        Returns:
            List of result dicts with:
                - id: Document ID
                - score: Similarity score
                - All payload fields (text, title, procedure_id, etc.)
        
        Raises:
            VectorStoreOperationError: If search operation fails
        """
        try:
            # Convert filter_dict to Qdrant filter format
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # List matching: check if any value in list matches
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=value))
                        )
                    else:
                        # Exact match
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Ensure query_embedding is 1D
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            # Use query_points (search was removed in qdrant-client 1.16+)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                query_filter=query_filter
            )
            results = response.points if hasattr(response, "points") else []
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": str(result.id),
                    "score": result.score,
                    **(result.payload or {})
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise VectorStoreOperationError(f"Search failed: {e}") from e
    
    def update(
        self,
        doc_id: Union[str, int],
        embedding: Optional[np.ndarray] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update document embedding or metadata.
        
        Args:
            doc_id: Document ID to update
            embedding: Optional new embedding vector
            payload: Optional new payload/metadata dict
        
        Raises:
            VectorStoreOperationError: If update operation fails
        """
        if embedding is None and payload is None:
            logger.warning(f"No updates provided for document {doc_id}")
            return
        
        try:
            if payload is not None:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[doc_id]
                )
            
            if embedding is not None:
                # Ensure embedding is 1D
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=doc_id, vector=embedding.tolist())]
                )
            
            logger.debug(f"Updated document {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise VectorStoreOperationError(
                f"Failed to update document {doc_id}: {e}"
            ) from e
    
    def update_document(
        self,
        doc_id: Union[str, int],
        embedding: Optional[np.ndarray] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update document embedding or metadata.
        
        .. deprecated:: Use update() instead.
        
        Args:
            doc_id: Document ID to update
            embedding: Optional new embedding vector
            payload: Optional new payload/metadata dict
        """
        import warnings
        warnings.warn(
            "update_document() is deprecated. Use update() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.update(doc_id, embedding, payload)
    
    def delete(
        self,
        doc_id: Optional[Union[str, int]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents from vector store.
        
        Either doc_id or filter_dict must be provided, but not both.
        
        Args:
            doc_id: Document ID to delete (single document)
            filter_dict: Metadata filter to delete matching documents
        
        Raises:
            VectorStoreOperationError: If delete operation fails or arguments are invalid
        """
        if doc_id is None and filter_dict is None:
            raise VectorStoreOperationError(
                "Either doc_id or filter_dict must be provided"
            )
        
        if doc_id is not None and filter_dict is not None:
            raise VectorStoreOperationError(
                "Cannot specify both doc_id and filter_dict"
            )
        
        try:
            if doc_id is not None:
                # Delete single document by ID
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[doc_id]
                )
                logger.info(f"Deleted document {doc_id}")
            else:
                # Delete by filter
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=value))
                        )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=query_filter
                    )
                    logger.info(f"Deleted documents matching filter: {filter_dict}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise VectorStoreOperationError(f"Delete failed: {e}") from e
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            Dict with collection metadata:
                - name: Collection name
                - vectors_count: Number of vectors
                - points_count: Number of points
                - config: Collection configuration
        
        Raises:
            VectorStoreOperationError: If operation fails
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": info.config.dict()
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise VectorStoreOperationError(
                f"Failed to get collection info: {e}"
            ) from e





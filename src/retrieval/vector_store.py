"""
Vector store interface for repair guide embeddings.

Uses ChromaDB Cloud (CHROMA_DB_API_KEY, CHROMA_DB_TENANT from .env).
"""
from typing import Dict, Any

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


def get_vector_store(config: Dict[str, Any]):
    """
    Return the ChromaDB vector store.

    Args:
        config: Vector store config (provider, collection_name, database, etc.)

    Returns:
        ChromaVectorStore instance
    """
    from .chroma_store import ChromaVectorStore
    return ChromaVectorStore(config)


# Primary export
VectorStore = get_vector_store

"""
Centralized path management for MIST project.
Provides single source of truth for all file paths with fallback support.
"""
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


class Paths:
    """Centralized path management for MIST project"""
    
    def __init__(self, mist_root=None):
        if mist_root is None:
            # Auto-detect mist root by finding this file's parent
            mist_root = Path(__file__).parent.parent
        self.mist_root = Path(mist_root).resolve()
        
    @property
    def config(self):
        """Configuration directory"""
        return self.mist_root / "config"
    
    @property
    def data(self):
        """Data directory"""
        return self.mist_root / "data"
    
    @property
    def databases(self):
        """Database directory"""
        # Check environment override first
        env_path = os.getenv("MIST_DATABASE_DIR")
        if env_path:
            return Path(env_path)
        
        # Use data/databases directory
        return self.data / "databases"
    
    @property
    def knowledge_graph(self):
        """Knowledge graph file path"""
        return self.data / "knowledge_graph.graphml"
    
    @property
    def vector_store(self):
        """Vector store directory"""
        env_path = os.getenv("MIST_VECTOR_STORE_DIR")
        if env_path:
            return Path(env_path)
        return self.data / "vector_store"
    
    @property
    def feedback_db(self):
        """Feedback database file path"""
        return self.data / "feedback" / "feedback.db"
    
    @property
    def embeddings_checkpoints(self):
        """Embedding checkpoint directory"""
        return self.data / "embeddings" / "checkpoints"
    
    def get_database_path(self, db_name):
        """
        Get path to a specific database file.
        
        Args:
            db_name: Name of database file (e.g., "DiagDocDb_DECRYPTED.sqlite")
        
        Returns:
            Path to database file in data/databases directory
        """
        return self.databases / db_name


# Global instance
_paths_instance = None


def get_paths():
    """Get global Paths instance"""
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = Paths()
    return _paths_instance

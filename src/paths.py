"""
Centralized path management for MIST project.
Provides single source of truth for all file paths with fallback support.

This module provides a Paths class that manages all file and directory paths
used throughout the MIST project. It supports environment variable overrides
and optional directory creation.
"""
from pathlib import Path
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class Paths:
    """
    Centralized path management for MIST project.
    
    Provides properties and methods to access all project paths with support
    for environment variable overrides and optional directory creation.
    
    Attributes:
        mist_root: Root directory of the MIST project
    """
    
    def __init__(self, mist_root: Optional[Path] = None):
        """
        Initialize Paths instance.
        
        Args:
            mist_root: Root directory of MIST project. If None, auto-detects
                      by finding this file's parent directory.
        """
        if mist_root is None:
            # Auto-detect mist root by finding this file's parent
            mist_root = Path(__file__).parent.parent
        self.mist_root = Path(mist_root).resolve()
    
    def __repr__(self) -> str:
        """Return string representation of Paths instance."""
        return f"Paths(mist_root={self.mist_root})"
    
    @property
    def config(self) -> Path:
        """
        Configuration directory.
        
        Can be overridden with MIST_CONFIG_DIR environment variable.
        
        Returns:
            Path to config directory
        """
        env_path = os.getenv("MIST_CONFIG_DIR")
        if env_path:
            path = Path(env_path).resolve()
            logger.info(f"Using config dir from env: {path}")
            return path
        path = self.mist_root / "config"
        logger.info(f"Using default config dir: {path}")
        return path
    
    @property
    def data(self) -> Path:
        """Data directory."""
        return self.mist_root / "data"
    
    @property
    def src(self) -> Path:
        """Source code directory."""
        return self.mist_root / "src"
    
    @property
    def scripts(self) -> Path:
        """
        Scripts directory.
        
        Can be overridden with MIST_SCRIPTS_DIR environment variable.
        
        Returns:
            Path to scripts directory
        """
        env_path = os.getenv("MIST_SCRIPTS_DIR")
        if env_path:
            return Path(env_path).resolve()
        return self.mist_root / "scripts"
    
    @property
    def migrations(self) -> Path:
        """Migrations directory (scripts/migrations)."""
        return self.scripts / "migrations"
    
    @property
    def tests(self) -> Path:
        """Tests directory."""
        return self.mist_root / "tests"
    
    @property
    def databases(self) -> Path:
        """
        Database directory.
        
        Can be overridden with MIST_DATABASE_DIR environment variable.
        
        Returns:
            Path to databases directory
        """
        env_path = os.getenv("MIST_DATABASE_DIR")
        if env_path:
            return Path(env_path).resolve()
        return self.data / "databases"
    
    @property
    def knowledge_graph(self) -> Path:
        """Knowledge graph file path."""
        return self.data / "knowledge_graph.graphml"
    
    @property
    def vector_store(self) -> Path:
        """
        Vector store directory.
        
        Can be overridden with MIST_VECTOR_STORE_DIR environment variable.
        
        Returns:
            Path to vector store directory
        """
        env_path = os.getenv("MIST_VECTOR_STORE_DIR")
        if env_path:
            return Path(env_path).resolve()
        return self.data / "vector_store"
    
    @property
    def feedback(self) -> Path:
        """Feedback directory."""
        return self.data / "feedback"
    
    @property
    def feedback_db(self) -> Path:
        """Feedback database file path."""
        return self.feedback / "feedback.db"
    
    @property
    def embeddings(self) -> Path:
        """Embeddings directory."""
        return self.data / "embeddings"
    
    @property
    def embeddings_checkpoints(self) -> Path:
        """Embedding checkpoint directory."""
        return self.embeddings / "checkpoints"
    
    def get_config_path(self, config_name: str) -> Path:
        """
        Get path to a configuration file.
        
        Args:
            config_name: Name of config file (e.g., "embedding_config.yaml")
        
        Returns:
            Path to config file in config directory
        """
        return self.config / config_name
    
    @property
    def embedding_config(self) -> Path:
        """Path to embedding configuration file."""
        return self.get_config_path("embedding_config.yaml")
    
    @property
    def llm_config(self) -> Path:
        """Path to LLM configuration file."""
        return self.get_config_path("llm_config.yaml")
    
    @property
    def retrieval_config(self) -> Path:
        """Path to retrieval configuration file."""
        return self.get_config_path("retrieval_config.yaml")
    
    @property
    def training_config(self) -> Path:
        """Path to training configuration file."""
        return self.get_config_path("training_config.yaml")
    
    def get_database_path(self, db_name: str) -> Path:
        """
        Get path to a specific database file.
        
        Args:
            db_name: Name of database file (e.g., "DiagDocDb_Decrypted.sqlite")
        
        Returns:
            Path to database file in data/databases directory
        """
        return self.databases / db_name

    def get_ista_db_path(self) -> Path:
        """
        Get path to BMW ISTA database, trying multiple locations.
        
        Tries: ISTA_DB_PATH env, then DiagDocDb_Decrypted.sqlite,
        then DiagDocDb_DECRYPTED.sqlite (case variations for WSL).
        
        Returns:
            Path to existing file, or None if not found
        """
        env_path = os.getenv("ISTA_DB_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()
        for name in ("DiagDocDb_Decrypted.sqlite", "DiagDocDb_DECRYPTED.sqlite"):
            p = self.databases / name
            if p.exists():
                return p
        return self.databases / "DiagDocDb_Decrypted.sqlite"  # Default for error message
    
    def get_mist_db_path(self) -> Path:
        """
        Get path to MIST database file (mist_data.db).
        
        Returns:
            Path to mist_data.db file
        """
        return self.databases / "mist_data.db"
    
    def get_migration_sql_path(self) -> Path:
        """
        Get path to migration SQL file.
        
        Returns:
            Path to create_mist_tables.sql file
        """
        return self.migrations / "create_mist_tables.sql"
    
    def ensure_directories(self, create_if_missing: bool = True) -> None:
        """
        Ensure all standard directories exist.
        
        Creates missing directories if create_if_missing is True.
        Logs warnings for missing directories if create_if_missing is False.
        
        Args:
            create_if_missing: If True, create missing directories. If False,
                             only log warnings.
        """
        directories = [
            self.config,
            self.data,
            self.src,
            self.scripts,
            self.migrations,
            self.tests,
            self.databases,
            self.vector_store,
            self.feedback,
            self.embeddings,
            self.embeddings_checkpoints,
        ]
        
        for directory in directories:
            if not directory.exists():
                if create_if_missing:
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {directory}")
                    except OSError as e:
                        logger.error(f"Failed to create directory {directory}: {e}")
                else:
                    logger.warning(f"Directory does not exist: {directory}")


# Global instance
_paths_instance = None


def get_paths():
    """Get global Paths instance"""
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = Paths()
    return _paths_instance

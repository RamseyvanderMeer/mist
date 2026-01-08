"""
Database module for MIST system.

Provides SQLAlchemy models and migration utilities for MIST database schema.
"""
from pathlib import Path
from typing import Optional
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker, Session

from .schema import (
    Base,
    FeedbackSession,
    MistEmbedding,
    MistFeedback,
    MistTrainingCheckpoint
)
from .migrations import (
    run_migrations,
    validate_schema,
    get_schema_version,
    create_engine_for_db,
    init_database
)

# Import paths module for database path resolution
try:
    from ..paths import get_paths
except ImportError:
    # Fallback if paths module not available
    def get_paths():
        class Paths:
            @property
            def databases(self):
                return Path(__file__).parent.parent.parent / "data" / "databases"
        return Paths()


def get_mist_db_path() -> Path:
    """
    Get path to MIST database file.
    
    Returns:
        Path to mist_data.db in data/databases directory
    """
    paths = get_paths()
    return paths.databases / "mist_data.db"


def get_mist_db_engine() -> Engine:
    """
    Get SQLAlchemy engine for MIST database.
    
    Returns:
        SQLAlchemy Engine instance
    """
    db_path = get_mist_db_path()
    return create_engine_for_db(str(db_path))


def get_mist_db_session() -> Session:
    """
    Get SQLAlchemy session for MIST database.
    
    Returns:
        SQLAlchemy Session instance
    """
    engine = get_mist_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def ensure_mist_database() -> bool:
    """
    Ensure MIST database exists and is initialized with schema.
    
    Returns:
        True if database is ready, False otherwise
    """
    db_path = get_mist_db_path()
    return init_database(str(db_path))


__all__ = [
    # Models
    'Base',
    'FeedbackSession',
    'MistEmbedding',
    'MistFeedback',
    'MistTrainingCheckpoint',
    # Migration utilities
    'run_migrations',
    'validate_schema',
    'get_schema_version',
    'create_engine_for_db',
    'init_database',
    # Convenience functions
    'get_mist_db_path',
    'get_mist_db_engine',
    'get_mist_db_session',
    'ensure_mist_database',
]

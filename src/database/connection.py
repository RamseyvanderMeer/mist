"""
Database connection management for MIST system.

Provides DatabaseConnection class for managing SQLAlchemy engine and sessions.
"""
import logging
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages SQLAlchemy database engine and session lifecycle.
    
    Provides context-managed sessions for safe database operations.
    """
    
    def __init__(self, engine: Engine):
        """
        Initialize with a SQLAlchemy engine.
        
        Args:
            engine: SQLAlchemy Engine instance
        """
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine)
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        return self._engine
    
    @contextmanager
    def session(self):
        """
        Provide a transactional scope around a series of operations.
        
        Yields:
            SQLAlchemy Session instance
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """Dispose of the engine and close all connections."""
        self._engine.dispose()


def create_connection(db_path, **kwargs) -> DatabaseConnection:
    """
    Create a DatabaseConnection for a SQLite database file.
    
    Args:
        db_path: Path to the SQLite database file
        **kwargs: Additional arguments passed to create_engine
    
    Returns:
        DatabaseConnection instance
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    connection_string = f"sqlite:///{db_path.absolute()}"
    
    engine = create_engine(
        connection_string,
        connect_args={"check_same_thread": False},
        echo=kwargs.get("echo", False),
    )
    
    return DatabaseConnection(engine)

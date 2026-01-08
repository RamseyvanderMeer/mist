"""
Database connection manager for SQLAlchemy.

Provides reusable connection management with context manager support
for automatic session cleanup and error handling.
"""
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    SQLAlchemy database connection manager with context manager support.
    
    Manages engine creation, session handling, and automatic cleanup.
    Provides thread-safe session management with proper error handling.
    
    Example:
        ```python
        conn = DatabaseConnection("sqlite:///path/to/db.sqlite")
        with conn.session() as session:
            result = session.execute(text("SELECT 1"))
        ```
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
        check_same_thread: bool = False
    ):
        """
        Initialize database connection manager.
        
        Args:
            connection_string: SQLAlchemy connection string
                               (e.g., "sqlite:///path/to/db.sqlite")
            pool_size: Connection pool size (default: 5)
            max_overflow: Maximum overflow connections (default: 10)
            echo: Enable SQL query logging (default: False)
            check_same_thread: For SQLite, allow multi-threaded access (default: False)
        """
        self.connection_string = connection_string
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
        # SQLite-specific connection args
        connect_args = {}
        if connection_string.startswith("sqlite"):
            connect_args["check_same_thread"] = check_same_thread
        
        # Create engine with connection pooling
        self._engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
            connect_args=connect_args
        )
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
        
        logger.debug(f"Initialized DatabaseConnection for {connection_string}")
    
    @property
    def engine(self) -> Engine:
        """
        Get SQLAlchemy engine.
        
        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized")
        return self._engine
    
    @contextmanager
    def session(self):
        """
        Context manager for database session.
        
        Automatically handles session creation, commit/rollback, and cleanup.
        
        Yields:
            SQLAlchemy Session instance
            
        Example:
            ```python
            with conn.session() as session:
                # Use session here
                result = session.execute(text("SELECT 1"))
                session.commit()
            ```
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error, rolling back: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error, rolling back: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """
        Get a new database session (manual management required).
        
        Note: Prefer using `session()` context manager for automatic cleanup.
        
        Returns:
            SQLAlchemy Session instance
            
        Warning:
            Caller is responsible for closing the session.
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")
        return self._session_factory()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            logger.debug("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """Close all database connections and dispose of engine."""
        if self._engine:
            self._engine.dispose()
            logger.debug("Database connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connections."""
        self.close()
        return False


def create_connection(
    db_path: str | Path,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False
) -> DatabaseConnection:
    """
    Create a DatabaseConnection instance from a file path.
    
    Convenience function for creating connections to SQLite databases.
    
    Args:
        db_path: Path to database file
        pool_size: Connection pool size (default: 5)
        max_overflow: Maximum overflow connections (default: 10)
        echo: Enable SQL query logging (default: False)
    
    Returns:
        DatabaseConnection instance
    """
    db_path = Path(db_path)
    
    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create SQLite connection string
    connection_string = f"sqlite:///{db_path.absolute()}"
    
    return DatabaseConnection(
        connection_string=connection_string,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
        check_same_thread=False
    )

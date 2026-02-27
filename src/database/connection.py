"""
Database connection management for MIST system.

Provides DatabaseConnection wrapper for SQLite databases with session context manager.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from .migrations import create_engine_for_db


class DatabaseConnection:
    """
    Wrapper for SQLite database connections with session management.

    Provides test_connection() and session() context manager for
    SQLAlchemy operations.
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._engine = create_engine_for_db(str(self.db_path))
        self._session_factory = sessionmaker(
            bind=self._engine, autocommit=False, autoflush=False
        )

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
        except Exception:
            return False

    @contextmanager
    def session(self):
        """
        Provide a transactional scope for database operations.

        Yields:
            SQLAlchemy Session instance. Commits on success, rolls back on exception.
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def create_connection(db_path: Union[str, Path]) -> DatabaseConnection:
    """
    Create a DatabaseConnection for the given SQLite database path.

    Args:
        db_path: Path to SQLite database file

    Returns:
        DatabaseConnection instance
    """
    return DatabaseConnection(db_path)

"""PostgreSQL connection and session management for auth/users."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Create engine with connection pooling
pg_engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Session factory
PGSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection."""
    db = PGSessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = PGSessionLocal()
    try:
        yield db
    finally:
        db.close()

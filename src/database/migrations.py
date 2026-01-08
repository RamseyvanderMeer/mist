"""
Database migration utilities for MIST schema.
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def run_migrations(db_path: str, migration_file: Optional[str] = None) -> bool:
    """
    Execute SQL migration script idempotently.
    
    Args:
        db_path: Path to SQLite database file
        migration_file: Path to SQL migration file. If None, uses default.
    
    Returns:
        True if migration succeeded, False otherwise
    """
    db_path = Path(db_path)
    
    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use default migration file if not provided
    if migration_file is None:
        migration_file = Path(__file__).parent.parent.parent / "scripts" / "migrations" / "create_mist_tables.sql"
    else:
        migration_file = Path(migration_file)
    
    if not migration_file.exists():
        logger.error(f"Migration file not found: {migration_file}")
        return False
    
    try:
        # Read migration SQL
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Execute migration (idempotent due to IF NOT EXISTS clauses)
        cursor.executescript(migration_sql)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Migration executed successfully on {db_path}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error during migration: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        return False


def validate_schema(db_path: str) -> Tuple[bool, List[str]]:
    """
    Verify all tables and indexes exist in the database.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        Tuple of (is_valid, list_of_missing_items)
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        return False, [f"Database file does not exist: {db_path}"]
    
    expected_tables = [
        'feedback_sessions',
        'mist_embeddings',
        'mist_feedback',
        'mist_training_checkpoints'
    ]
    
    expected_indexes = [
        'idx_mist_embeddings_procedure',
        'idx_mist_embeddings_version',
        'idx_mist_feedback_session',
        'idx_mist_feedback_procedure'
    ]
    
    missing_items = []
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        for table in expected_tables:
            if table not in existing_tables:
                missing_items.append(f"Table missing: {table}")
        
        # Check indexes
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        existing_indexes = {row[0] for row in cursor.fetchall()}
        
        for index in expected_indexes:
            if index not in existing_indexes:
                missing_items.append(f"Index missing: {index}")
        
        conn.close()
        
        is_valid = len(missing_items) == 0
        return is_valid, missing_items
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error during schema validation: {e}")
        return False, [f"Database error: {e}"]
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {e}")
        return False, [f"Validation error: {e}"]


def get_schema_version(db_path: str) -> Optional[int]:
    """
    Get current schema version from database.
    
    Note: This is a placeholder for future version tracking.
    Currently returns None as version tracking is not yet implemented.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        Schema version number or None if not tracked
    """
    # Future implementation: Add schema_version table
    # For now, return None
    return None


def create_engine_for_db(db_path: str) -> Engine:
    """
    Create SQLAlchemy engine for database.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        SQLAlchemy Engine instance
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # SQLite connection string
    connection_string = f"sqlite:///{db_path.absolute()}"
    
    return create_engine(
        connection_string,
        connect_args={"check_same_thread": False},  # Allow multi-threaded access
        echo=False  # Set to True for SQL query logging
    )


def init_database(db_path: str, migration_file: Optional[str] = None) -> bool:
    """
    Initialize database with schema if it doesn't exist or is invalid.
    
    Args:
        db_path: Path to SQLite database file
        migration_file: Optional path to migration file
    
    Returns:
        True if initialization succeeded, False otherwise
    """
    db_path = Path(db_path)
    
    # Check if database exists and is valid
    if db_path.exists():
        is_valid, missing = validate_schema(str(db_path))
        if is_valid:
            logger.info(f"Database {db_path} already exists and is valid")
            return True
        else:
            logger.warning(f"Database {db_path} exists but schema is invalid: {missing}")
    
    # Run migrations
    return run_migrations(str(db_path), migration_file)

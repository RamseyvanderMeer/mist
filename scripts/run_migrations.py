#!/usr/bin/env python3
"""
Script to run MIST database migrations.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import ensure_mist_database, get_mist_db_path, validate_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run migrations."""
    logger.info("Running MIST database migrations...")
    
    # Get database path
    db_path = get_mist_db_path()
    logger.info(f"Database path: {db_path}")
    
    # Run migrations
    success = ensure_mist_database()
    
    if success:
        logger.info("✓ Migrations completed successfully!")
        
        # Validate schema
        is_valid, missing = validate_schema(str(db_path))
        if is_valid:
            logger.info("✓ Schema validation passed")
            return 0
        else:
            logger.error(f"✗ Schema validation failed: {missing}")
            return 1
    else:
        logger.error("✗ Migrations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

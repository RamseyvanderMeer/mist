"""Run MIST database migrations."""
import logging

from src.database import ensure_mist_database, get_mist_db_path, validate_schema

logger = logging.getLogger(__name__)


def run() -> int:
    """Run migrations. Returns 0 on success, 1 on failure."""
    logger.info("Running MIST database migrations...")
    db_path = get_mist_db_path()
    logger.info("Database path: %s", db_path)

    success = ensure_mist_database()
    if not success:
        logger.error("Migrations failed")
        return 1

    logger.info("Migrations completed successfully!")
    is_valid, missing = validate_schema(str(db_path))
    if is_valid:
        logger.info("Schema validation passed")
        return 0
    logger.error("Schema validation failed: %s", missing)
    return 1

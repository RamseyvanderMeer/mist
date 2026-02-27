"""
Fetch full procedure content from xmlvalueprimitive_ENGB database.

Uses FTS search by title to find XML content, strips tags, returns full text.
Per ISTA_DATABASE_GUIDE: xmlvalueprimitive contains actual text content (XML).
"""
import re
import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class XmlContentFetcher:
    """
    Fetches procedure content from xmlvalueprimitive database via FTS search.
    
    The xml db stores XML; we strip tags and return plain text for embedding.
    """

    FTS_TABLE = "fts"
    CONTENT_COL = "data"

    def __init__(self, xml_db_path: Optional[Path] = None):
        """
        Initialize fetcher.
        
        Args:
            xml_db_path: Path to xmlvalueprimitive_ENGB.sqlite. If None, tries
                         paths.get_database_path().
        """
        self._conn: Optional[sqlite3.Connection] = None
        self._path: Optional[Path] = None
        
        if xml_db_path is None:
            try:
                from ..paths import get_paths
                paths = get_paths()
                for name in ("xmlvalueprimitive_ENGB.sqlite", "xmlvalueprimitive_ENGB_complete.sqlite"):
                    p = paths.get_database_path(name)
                    if p and p.exists():
                        xml_db_path = p
                        break
            except ImportError:
                pass
        
        if xml_db_path and Path(xml_db_path).exists():
            self._path = Path(xml_db_path)
            try:
                self._conn = sqlite3.connect(str(self._path))
                self._conn.row_factory = sqlite3.Row
                logger.info(f"XmlContentFetcher: connected to {self._path.name}")
            except Exception as e:
                logger.warning(f"XmlContentFetcher: could not connect to {xml_db_path}: {e}")
                self._conn = None

    def get_content(self, procedure_id: str, title: str) -> Optional[str]:
        """
        Get full procedure content by FTS search on title.
        
        Args:
            procedure_id: Procedure ID (for logging)
            title: Procedure title used for FTS search
            
        Returns:
            Full plain text content or None if not found
        """
        if not self._conn or not title or not title.strip():
            return None
        
        try:
            cursor = self._conn.cursor()
            # Build FTS query from first 3–5 significant words
            terms = [t for t in title.split() if len(t) > 2][:5]
            if not terms:
                return None
            search_query = " OR ".join(f'"{t}"' for t in terms)
            
            cursor.execute(
                f"SELECT {self.CONTENT_COL} FROM {self.FTS_TABLE} WHERE {self.FTS_TABLE} MATCH ? LIMIT 1",
                (search_query,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            raw = row[0]
            if not raw:
                return None
            
            text = self._strip_xml(str(raw))
            if len(text) > 50:
                return text
            return None
        except sqlite3.OperationalError as e:
            logger.debug(f"XmlContentFetcher FTS failed for {procedure_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"XmlContentFetcher error for {procedure_id}: {e}")
            return None

    def _strip_xml(self, xml: str) -> str:
        """Remove XML tags and normalize whitespace."""
        text = re.sub(r"<[^>]+>", " ", xml)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "XmlContentFetcher":
        return self

    def __exit__(self, *args) -> None:
        self.close()

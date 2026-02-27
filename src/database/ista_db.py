"""
BMW ISTA database wrapper for querying diagnostic databases.

Provides high-level query interface for fault codes, ECUs, repair procedures,
and diagnostic objects from BMW ISTA databases.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session

from .connection import DatabaseConnection, create_connection
from .fault_code_mapping import get_lookup_variants

logger = logging.getLogger(__name__)


class IstaDatabase:
    """
    High-level wrapper for BMW ISTA diagnostic database queries.
    
    Provides typed query methods for accessing fault codes, ECUs,
    repair procedures, and diagnostic objects.
    
    Example:
        ```python
        ista_db = IstaDatabase()
        
        # Get fault code information
        fault = ista_db.get_fault_code("P0301")
        
        # Get repair procedures for a fault
        procedures = ista_db.get_procedures_for_fault("P0301")
        
        # Search procedures
        results = ista_db.search_procedures("oil change", limit=10)
        ```
    """
    
    DEFAULT_DB_NAME = "DiagDocDb_Decrypted.sqlite"
    
    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        connection: Optional[DatabaseConnection] = None
    ):
        """
        Initialize ISTA database wrapper.
        
        Args:
            db_path: Path to ISTA database file. If None, uses default from paths module.
            connection: Optional DatabaseConnection instance. If provided, uses this
                       instead of creating a new connection.
        """
        if connection is not None:
            self._connection = connection
            self._db_path = None
        else:
            if db_path is None:
                # Use paths module to get default database path
                try:
                    from ..paths import get_paths
                    paths = get_paths()
                    db_path = paths.get_database_path(self.DEFAULT_DB_NAME)
                except ImportError:
                    # Fallback if paths module not available
                    db_path = Path(__file__).parent.parent.parent / "data" / "databases" / self.DEFAULT_DB_NAME
            
            self._db_path = Path(db_path)
            self._connection = create_connection(self._db_path)
        
        logger.info(f"Initialized IstaDatabase with database: {self._db_path or 'provided connection'}")
    
    @property
    def connection(self) -> DatabaseConnection:
        """Get database connection instance."""
        return self._connection
    
    def get_database_path(self) -> Optional[Path]:
        """
        Get path to ISTA database file.
        
        Returns:
            Path to database file, or None if using provided connection
        """
        return self._db_path
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        return self._connection.test_connection()
    
    # Fault Code Methods
    
    def get_fault_code(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Get fault code information by code.
        
        Args:
            code: Fault code (e.g., "P0301")
        
        Returns:
            Dictionary with fault code information, or None if not found
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("SELECT * FROM XEP_FAULTCODES WHERE CODE = :code LIMIT 1"),
                    {"code": code}
                )
                row = result.fetchone()
                
                if row is None:
                    return None
                
                # Convert row to dictionary (SQLAlchemy 2.0 compatible)
                if hasattr(row, '_mapping'):
                    return dict(row._mapping)
                elif hasattr(row, '_asdict'):
                    return row._asdict()
                else:
                    # Fallback: create dict from row keys and values
                    return {key: getattr(row, key) for key in row.keys()}
        except Exception as e:
            logger.error(f"Error querying fault code {code}: {e}")
            raise
    
    def get_all_fault_codes(
        self,
        code_pattern: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get all distinct fault codes from the database.
        
        Args:
            code_pattern: Optional SQL LIKE pattern to filter codes (e.g., "P%" for
                         standard OBD-II P-codes only). If None, returns all codes.
            limit: Optional maximum number of codes to return. If None, returns all.
        
        Returns:
            List of fault code strings (e.g., ["P0300", "P0301", "2A87"])
        """
        try:
            with self._connection.session() as session:
                sql = "SELECT DISTINCT CODE FROM XEP_FAULTCODES WHERE CODE IS NOT NULL AND CODE != ''"
                params: Dict[str, Any] = {}
                if code_pattern:
                    sql += " AND CODE LIKE :pattern"
                    params["pattern"] = code_pattern
                sql += " ORDER BY CODE"
                if limit:
                    sql += " LIMIT :limit"
                    params["limit"] = limit
                result = session.execute(text(sql), params)
                rows = result.fetchall()
                codes = []
                for row in rows:
                    code = row[0] if isinstance(row, tuple) else getattr(row, "CODE", None)
                    if code:
                        codes.append(str(code))
                return codes
        except Exception as e:
            logger.error(f"Error fetching fault codes: {e}")
            raise
    
    def get_fault_labels(self, fault_code_id: str) -> List[Dict[str, Any]]:
        """
        Get fault code labels/descriptions by fault code ID.
        
        Args:
            fault_code_id: Fault code ID from XEP_FAULTCODES table
        
        Returns:
            List of dictionaries with fault label information
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("SELECT * FROM XEP_FAULTLABELS WHERE FAULTCODE_ID = :id"),
                    {"id": fault_code_id}
                )
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error querying fault labels for ID {fault_code_id}: {e}")
            raise
    
    def search_fault_codes(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search fault codes by code or description.
        
        Args:
            query: Search query (searches CODE and TITLE_ENGB fields)
            limit: Maximum number of results (default: 100)
        
        Returns:
            List of dictionaries with fault code information
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("""
                        SELECT DISTINCT fc.*
                        FROM XEP_FAULTCODES fc
                        LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.FAULTCODE_ID
                        WHERE fc.CODE LIKE :query
                           OR fl.LABEL LIKE :query
                           OR fl.DESCRIPTION LIKE :query
                        LIMIT :limit
                    """),
                    {"query": f"%{query}%", "limit": limit}
                )
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error searching fault codes with query '{query}': {e}")
            raise
    
    # ECU Methods
    
    def get_ecu_variants(self, type_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get ECU variants, optionally filtered by type key.
        
        Args:
            type_key: Optional vehicle type key to filter results
        
        Returns:
            List of dictionaries with ECU variant information
        """
        try:
            with self._connection.session() as session:
                if type_key:
                    result = session.execute(
                        text("SELECT * FROM XEP_ECUVARIANTS WHERE TYPEKEY = :type_key"),
                        {"type_key": type_key}
                    )
                else:
                    result = session.execute(
                        text("SELECT * FROM XEP_ECUVARIANTS LIMIT 1000")
                    )
                
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error querying ECU variants: {e}")
            raise
    
    def get_ecu_by_id(self, ecu_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ECU variant by ID.
        
        Args:
            ecu_id: ECU variant ID
        
        Returns:
            Dictionary with ECU information, or None if not found
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("SELECT * FROM XEP_ECUVARIANTS WHERE ID = :id LIMIT 1"),
                    {"id": ecu_id}
                )
                row = result.fetchone()
                
                if row is None:
                    return None
                
                return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        except Exception as e:
            logger.error(f"Error querying ECU by ID {ecu_id}: {e}")
            raise
    
    def get_ecu_groups(self) -> List[Dict[str, Any]]:
        """
        Get ECU groups.
        
        Returns:
            List of dictionaries with ECU group information
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("SELECT * FROM XEP_ECUGROUPS LIMIT 1000")
                )
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error querying ECU groups: {e}")
            raise
    
    # Repair Procedure Methods
    
    def get_info_object(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """
        Get repair procedure metadata by ID.
        
        Args:
            procedure_id: Procedure ID from XEP_INFOOBJECTS table
        
        Returns:
            Dictionary with procedure metadata, or None if not found
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("SELECT * FROM XEP_INFOOBJECTS WHERE ID = :id LIMIT 1"),
                    {"id": procedure_id}
                )
                row = result.fetchone()
                
                if row is None:
                    return None
                
                return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        except Exception as e:
            logger.error(f"Error querying info object {procedure_id}: {e}")
            raise
    
    def get_info_segments(self, procedure_id: str) -> List[Dict[str, Any]]:
        """
        Get content segments for a repair procedure.
        
        Note: XEP_INFOSEGMENTS doesn't have a direct INFOOBJECTID column.
        Segments may be linked via CONTROLID or accessed through xmlvalueprimitive database.
        For now, this returns an empty list as segments aren't directly queryable.
        
        Args:
            procedure_id: Procedure ID from XEP_INFOOBJECTS table
        
        Returns:
            List of dictionaries with procedure segment information (empty for now)
        """
        try:
            # XEP_INFOSEGMENTS doesn't have INFOOBJECTID column
            # Segments are typically accessed via xmlvalueprimitive database using CONTENTID
            # For now, return empty list - the procedure title will be used for indexing
            logger.debug(f"Segments not directly queryable for procedure {procedure_id}, using title only")
            return []
        except Exception as e:
            logger.error(f"Error querying info segments for procedure {procedure_id}: {e}")
            # Return empty list instead of raising to allow script to continue
            return []
    
    def get_fault_codes_for_procedure(self, procedure_id: str) -> List[str]:
        """
        Get fault codes associated with a repair procedure.
        
        Args:
            procedure_id: Procedure ID from XEP_INFOOBJECTS table
        
        Returns:
            List of fault code strings (e.g., ["P0301", "P0302"])
        """
        try:
            with self._connection.session() as session:
                # RG_ECUFAULT_DOCIDS uses ECUFAULT_ID (not FAULTCODE_ID) and INFOOBJECTID (not DOCID)
                result = session.execute(
                    text("""
                        SELECT DISTINCT fc.CODE
                        FROM XEP_FAULTCODES fc
                        INNER JOIN RG_ECUFAULT_DOCIDS rg ON fc.ID = rg.ECUFAULT_ID
                        WHERE rg.INFOOBJECTID = :procedure_id
                        AND fc.CODE IS NOT NULL
                    """),
                    {"procedure_id": procedure_id}
                )
                rows = result.fetchall()
                
                fault_codes = []
                for row in rows:
                    code = row[0] if isinstance(row, tuple) else row.CODE
                    if code:
                        fault_codes.append(str(code))
                
                return fault_codes
        except Exception as e:
            logger.error(f"Error querying fault codes for procedure {procedure_id}: {e}")
            raise

    def get_fault_labels_for_procedure(self, procedure_id: str) -> List[str]:
        """
        Get human-readable fault labels/descriptions for a repair procedure.

        Returns problem/symptom descriptions (e.g. "Engine Coolant Temperature Sensor
        Circuit Malfunction") that help semantic search match user symptoms to procedures.

        Args:
            procedure_id: Procedure ID from XEP_INFOOBJECTS table

        Returns:
            List of label strings (TITLE_ENGB, LABEL, or DESCRIPTION from XEP_FAULTLABELS)
        """
        try:
            with self._connection.session() as session:
                # Try FAULTCODE_ID join (common schema); fallback to ID=ID
                for join_clause in [
                    "LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.FAULTCODE_ID",
                    "LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.ID",
                ]:
                    try:
                        result = session.execute(
                            text(f"""
                                SELECT DISTINCT fc.CODE, fl.TITLE_ENGB
                                FROM XEP_FAULTCODES fc
                                INNER JOIN RG_ECUFAULT_DOCIDS rg ON fc.ID = rg.ECUFAULT_ID
                                {join_clause}
                                WHERE rg.INFOOBJECTID = :procedure_id
                                AND fc.CODE IS NOT NULL
                            """),
                            {"procedure_id": procedure_id},
                        )
                        rows = result.fetchall()
                        labels = []
                        for row in rows:
                            label = None
                            if hasattr(row, "_mapping"):
                                label = row._mapping.get("TITLE_ENGB")
                            elif hasattr(row, "_asdict"):
                                label = row._asdict().get("TITLE_ENGB")
                            else:
                                label = row[1] if len(row) > 1 else None
                            if label and str(label).strip():
                                labels.append(str(label).strip())
                        if labels:
                            return labels
                    except Exception:
                        continue
                return []
        except Exception as e:
            logger.debug(f"Could not get fault labels for procedure {procedure_id}: {e}")
            return []

    def get_procedures_for_fault(self, fault_code: str) -> List[Dict[str, Any]]:
        """
        Get repair procedures linked to a fault code.
        
        Tries OBD-II P-code and BMW ISTA variants (e.g. P0015 -> 2A87).
        
        Args:
            fault_code: Fault code (e.g., "P0301" or "2A87")
        
        Returns:
            List of dictionaries with procedure information
        """
        try:
            variants = get_lookup_variants(fault_code)
            with self._connection.session() as session:
                fault_row = None
                for code in variants:
                    fault_result = session.execute(
                        text("SELECT ID FROM XEP_FAULTCODES WHERE CODE = :code LIMIT 1"),
                        {"code": code}
                    )
                    fault_row = fault_result.fetchone()
                    if fault_row is not None:
                        break

                if fault_row is None:
                    logger.warning(f"Fault code {fault_code} not found (tried: {variants})")
                    return []
                
                fault_id = fault_row[0]
                
                # Query RG_ECUFAULT_DOCIDS for linked procedures
                # Schema varies: ECUFAULT_ID/INFOOBJECTID (docs) vs FAULTCODE_ID/DOCID (tests)
                for fault_col, proc_col in [
                    ("ECUFAULT_ID", "INFOOBJECTID"),
                    ("ECUFAULT_ID", "DOCID"),
                    ("FAULTCODE_ID", "INFOOBJECTID"),
                    ("FAULTCODE_ID", "DOCID"),
                ]:
                    try:
                        result = session.execute(
                            text(f"""
                                SELECT io.*
                                FROM XEP_INFOOBJECTS io
                                INNER JOIN RG_ECUFAULT_DOCIDS rg ON io.ID = rg.{proc_col}
                                WHERE rg.{fault_col} = :fault_id
                            """),
                            {"fault_id": fault_id},
                        )
                        rows = result.fetchall()
                        if rows:
                            result_list = []
                            for row in rows:
                                if hasattr(row, "_mapping"):
                                    result_list.append(dict(row._mapping))
                                elif hasattr(row, "_asdict"):
                                    result_list.append(row._asdict())
                                else:
                                    result_list.append(
                                        {key: getattr(row, key) for key in row.keys()}
                                    )
                            return result_list
                    except Exception:
                        continue
                return []
        except Exception as e:
            logger.error(f"Error querying procedures for fault {fault_code}: {e}")
            raise
    
    def search_procedures(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search repair procedures by title.
        
        Args:
            query: Search query (searches TITLE_ENGB field)
            limit: Maximum number of results (default: 100)
        
        Returns:
            List of dictionaries with procedure information
        """
        try:
            with self._connection.session() as session:
                result = session.execute(
                    text("""
                        SELECT * FROM XEP_INFOOBJECTS
                        WHERE TITLE_ENGB LIKE :query
                        LIMIT :limit
                    """),
                    {"query": f"%{query}%", "limit": limit}
                )
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error searching procedures with query '{query}': {e}")
            raise
    
    # Diagnostic Object Methods
    
    def get_diagnosis_objects(self, fault_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get diagnostic objects, optionally filtered by fault code.
        
        Args:
            fault_code: Optional fault code to filter results
        
        Returns:
            List of dictionaries with diagnostic object information
        """
        try:
            with self._connection.session() as session:
                if fault_code:
                    # Get fault code ID first
                    fault_result = session.execute(
                        text("SELECT ID FROM XEP_FAULTCODES WHERE CODE = :code LIMIT 1"),
                        {"code": fault_code}
                    )
                    fault_row = fault_result.fetchone()
                    
                    if fault_row is None:
                        logger.warning(f"Fault code {fault_code} not found")
                        return []
                    
                    fault_id = fault_row[0]
                    
                    # Query diagnostic objects linked to fault
                    result = session.execute(
                        text("""
                            SELECT DISTINCT do.*
                            FROM XEP_DIAGNOSISOBJECTS do
                            INNER JOIN XEP_REFDIAGOBJECTS rdo ON do.ID = rdo.DIAGNOSISOBJECTID
                            WHERE rdo.FAULTCODE_ID = :fault_id
                        """),
                        {"fault_id": fault_id}
                    )
                else:
                    result = session.execute(
                        text("SELECT * FROM XEP_DIAGNOSISOBJECTS LIMIT 1000")
                    )
                
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error querying diagnosis objects: {e}")
            raise
    
    def get_fault_diagnostic_links(self, fault_code: str) -> List[Dict[str, Any]]:
        """
        Get fault-diagnostic relationships for a fault code.
        
        Args:
            fault_code: Fault code (e.g., "P0301")
        
        Returns:
            List of dictionaries with fault-diagnostic link information
        """
        try:
            with self._connection.session() as session:
                # Get fault code ID
                fault_result = session.execute(
                    text("SELECT ID FROM XEP_FAULTCODES WHERE CODE = :code LIMIT 1"),
                    {"code": fault_code}
                )
                fault_row = fault_result.fetchone()
                
                if fault_row is None:
                    logger.warning(f"Fault code {fault_code} not found")
                    return []
                
                fault_id = fault_row[0]
                
                # Query reference diagnostic objects
                result = session.execute(
                    text("SELECT * FROM XEP_REFDIAGOBJECTS WHERE FAULTCODE_ID = :fault_id"),
                    {"fault_id": fault_id}
                )
                rows = result.fetchall()
                
                # Convert rows to dictionaries (SQLAlchemy 2.0 compatible)
                result_list = []
                for row in rows:
                    if hasattr(row, '_mapping'):
                        result_list.append(dict(row._mapping))
                    elif hasattr(row, '_asdict'):
                        result_list.append(row._asdict())
                    else:
                        # Fallback: create dict from row keys and values
                        result_list.append({key: getattr(row, key) for key in row.keys()})
                return result_list
        except Exception as e:
            logger.error(f"Error querying fault-diagnostic links for {fault_code}: {e}")
            raise
    
    # Utility Methods
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table schema information.
        
        Args:
            table_name: Name of table to inspect
        
        Returns:
            List of dictionaries with column information
        """
        try:
            inspector = inspect(self._connection.engine)
            columns = inspector.get_columns(table_name)
            
            return [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default"),
                    "primary_key": col.get("primary_key", False)
                }
                for col in columns
            ]
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        self._connection.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connections."""
        self.close()
        return False

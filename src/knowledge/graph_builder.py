"""
Knowledge graph builder extracting relationships from BMW diagnostic database.
"""
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from sqlalchemy import text

from ..database.ista_db import IstaDatabase

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds knowledge graph from BMW diagnostic database relationships.
    
    Extracts nodes (fault codes, ECUs, diagnostic objects, repair procedures)
    and edges (affects_ecu, has_diagnostic, has_repair) from BMW ISTA databases.
    """
    
    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        ista_db: Optional[IstaDatabase] = None,
        incremental: bool = False
    ):
        """
        Initialize graph builder.
        
        Args:
            db_path: Path to BMW diagnostic database. Ignored if ista_db is provided.
            ista_db: Optional IstaDatabase instance. If provided, uses this instead of creating new.
            incremental: If True, merge with existing graph. If False, start fresh.
        """
        if ista_db is not None:
            self.ista_db = ista_db
        else:
            self.ista_db = IstaDatabase(db_path)
        
        self.graph = nx.MultiDiGraph()
        self.incremental = incremental
        
        # Statistics
        self.stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'errors': 0
        }
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            with self.ista_db.connection.session() as session:
                result = session.execute(
                    text("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=:table_name
                    """),
                    {"table_name": table_name}
                )
                return result.fetchone() is not None
        except Exception as e:
            logger.warning(f"Error checking if table {table_name} exists: {e}")
            return False
    
    def build(self) -> nx.MultiDiGraph:
        """
        Build knowledge graph from database.
        
        Returns:
            NetworkX MultiDiGraph
        """
        logger.info("Starting knowledge graph construction...")
        
        if not self.incremental:
            self.graph.clear()
            logger.info("Starting fresh graph (non-incremental mode)")
        
        try:
            # Extract nodes first
            logger.info("Extracting nodes...")
            self._extract_fault_codes()
            self._extract_ecus()
            self._extract_diagnostic_objects()
            self._extract_repair_procedures()
            
            # Extract relationships/edges
            logger.info("Extracting relationships...")
            self._extract_fault_ecu_relationships()
            self._extract_fault_diagnostic_relationships()
            self._extract_fault_repair_relationships()
            self._extract_diagnostic_tree_relationships()
            
            # Add graph metadata
            # Note: GraphML doesn't support dict values, so convert stats to string
            self.graph.graph['created_at'] = datetime.now().isoformat()
            self.graph.graph['nodes_count'] = self.graph.number_of_nodes()
            self.graph.graph['edges_count'] = self.graph.number_of_edges()
            self.graph.graph['stats'] = str(self.stats.copy())
            
            logger.info(
                f"Built knowledge graph with {self.graph.number_of_nodes()} nodes "
                f"and {self.graph.number_of_edges()} edges"
            )
            logger.info(
                f"Statistics: {self.stats['nodes_added']} nodes added, "
                f"{self.stats['edges_added']} edges added, "
                f"{self.stats['errors']} errors encountered"
            )
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}", exc_info=True)
            raise
        
        return self.graph
    
    def _extract_fault_codes(self):
        """Extract fault code nodes from XEP_FAULTCODES and XEP_FAULTLABELS tables."""
        if not self._table_exists("XEP_FAULTCODES"):
            logger.info("XEP_FAULTCODES table not found, skipping fault code extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # Query fault codes with labels
                # XEP_FAULTLABELS.ID matches XEP_FAULTCODES.ID
                result = session.execute(
                    text("""
                        SELECT DISTINCT fc.ID, fc.CODE, fl.TITLE_ENGB
                        FROM XEP_FAULTCODES fc
                        LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.ID
                        WHERE fc.CODE IS NOT NULL AND fc.CODE != ''
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by index or attribute
                        if hasattr(row, '_mapping'):
                            fault_id = str(row._mapping.get('ID', '')) if row._mapping.get('ID') else None
                            code = str(row._mapping.get('CODE', '')) if row._mapping.get('CODE') else None
                            title_engb = str(row._mapping.get('TITLE_ENGB', '')) if row._mapping.get('TITLE_ENGB') else ""
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            fault_id = str(row_dict.get('ID', '')) if row_dict.get('ID') else None
                            code = str(row_dict.get('CODE', '')) if row_dict.get('CODE') else None
                            title_engb = str(row_dict.get('TITLE_ENGB', '')) if row_dict.get('TITLE_ENGB') else ""
                        else:
                            # Fallback: access by index
                            fault_id = str(row[0]) if len(row) > 0 and row[0] else None
                            code = str(row[1]) if len(row) > 1 and row[1] else None
                            title_engb = str(row[2]) if len(row) > 2 and row[2] else ""
                        
                        if not fault_id or not code:
                            continue
                        
                        node_id = f"fault_code:{code}"
                        
                        # Skip if node already exists (incremental mode)
                        if node_id in self.graph:
                            continue
                        
                        # Add node with attributes
                        self.graph.add_node(
                            node_id,
                            node_type="fault_code",
                            id=fault_id,
                            code=code,
                            title_engb=title_engb or "",
                        )
                        count += 1
                        self.stats['nodes_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing fault code row: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} fault code nodes")
        except Exception as e:
            logger.error(f"Error extracting fault codes: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_ecus(self):
        """Extract ECU nodes from XEP_ECUVARIANTS and XEP_ECUGROUPS tables."""
        if not self._table_exists("XEP_ECUVARIANTS"):
            logger.info("XEP_ECUVARIANTS table not found, skipping ECU extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # Query ECU variants
                result = session.execute(
                    text("""
                        SELECT DISTINCT ev.ID, ev.NAME, ev.TITLE_ENGB
                        FROM XEP_ECUVARIANTS ev
                        WHERE ev.ID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by index or attribute
                        if hasattr(row, '_mapping'):
                            ecu_id = str(row._mapping.get('ID', '')) if row._mapping.get('ID') else None
                            name = str(row._mapping.get('NAME', '')) if row._mapping.get('NAME') else ""
                            title_engb = str(row._mapping.get('TITLE_ENGB', '')) if row._mapping.get('TITLE_ENGB') else ""
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            ecu_id = str(row_dict.get('ID', '')) if row_dict.get('ID') else None
                            name = str(row_dict.get('NAME', '')) if row_dict.get('NAME') else ""
                            title_engb = str(row_dict.get('TITLE_ENGB', '')) if row_dict.get('TITLE_ENGB') else ""
                        else:
                            # Fallback: access by index
                            ecu_id = str(row[0]) if len(row) > 0 and row[0] else None
                            name = str(row[1]) if len(row) > 1 and row[1] else ""
                            title_engb = str(row[2]) if len(row) > 2 and row[2] else ""
                        
                        if not ecu_id:
                            continue
                        
                        node_id = f"ecu:{ecu_id}"
                        
                        # Skip if node already exists
                        if node_id in self.graph:
                            continue
                        
                        self.graph.add_node(
                            node_id,
                            node_type="ecu",
                            id=ecu_id,
                            name=name or title_engb or "",
                        )
                        count += 1
                        self.stats['nodes_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing ECU row: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} ECU nodes")
        except Exception as e:
            logger.error(f"Error extracting ECUs: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_diagnostic_objects(self):
        """Extract diagnostic object nodes from XEP_DIAGNOSISOBJECTS table."""
        if not self._table_exists("XEP_DIAGNOSISOBJECTS"):
            logger.info("XEP_DIAGNOSISOBJECTS table not found, skipping diagnostic object extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # Query diagnostic objects
                result = session.execute(
                    text("""
                        SELECT DISTINCT do.ID, do.TITLE_ENGB, do.NAME, do.CONTROLID
                        FROM XEP_DIAGNOSISOBJECTS do
                        WHERE do.ID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by index or attribute
                        if hasattr(row, '_mapping'):
                            diag_id = str(row._mapping.get('ID', '')) if row._mapping.get('ID') else None
                            title_engb = str(row._mapping.get('TITLE_ENGB', '')) if row._mapping.get('TITLE_ENGB') else ""
                            name = str(row._mapping.get('NAME', '')) if row._mapping.get('NAME') else ""
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            diag_id = str(row_dict.get('ID', '')) if row_dict.get('ID') else None
                            title_engb = str(row_dict.get('TITLE_ENGB', '')) if row_dict.get('TITLE_ENGB') else ""
                            name = str(row_dict.get('NAME', '')) if row_dict.get('NAME') else ""
                        else:
                            # Fallback: access by index
                            diag_id = str(row[0]) if len(row) > 0 and row[0] else None
                            title_engb = str(row[1]) if len(row) > 1 and row[1] else ""
                            name = str(row[2]) if len(row) > 2 and row[2] else ""
                        
                        if not diag_id:
                            continue
                        
                        node_id = f"diagnostic:{diag_id}"
                        
                        # Skip if node already exists
                        if node_id in self.graph:
                            continue
                        
                        title = title_engb or name or ""
                        
                        self.graph.add_node(
                            node_id,
                            node_type="diagnostic",
                            id=diag_id,
                            title=title,
                        )
                        count += 1
                        self.stats['nodes_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing diagnostic object row: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} diagnostic object nodes")
        except Exception as e:
            logger.error(f"Error extracting diagnostic objects: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_repair_procedures(self):
        """Extract repair procedure nodes from XEP_INFOOBJECTS table."""
        if not self._table_exists("XEP_INFOOBJECTS"):
            logger.info("XEP_INFOOBJECTS table not found, skipping repair procedure extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # Query repair procedures
                result = session.execute(
                    text("""
                        SELECT DISTINCT io.ID, io.TITLE_ENGB, io.NAME
                        FROM XEP_INFOOBJECTS io
                        WHERE io.ID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        proc_id = str(row.ID) if row.ID else None
                        if not proc_id:
                            continue
                        
                        node_id = f"procedure:{proc_id}"
                        
                        # Skip if node already exists
                        if node_id in self.graph:
                            continue
                        
                        title_engb = str(row.TITLE_ENGB) if row.TITLE_ENGB else ""
                        name = str(row.NAME) if row.NAME else ""
                        
                        self.graph.add_node(
                            node_id,
                            node_type="procedure",
                            id=proc_id,
                            title_engb=title_engb or name,
                        )
                        count += 1
                        self.stats['nodes_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing repair procedure row: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} repair procedure nodes")
        except Exception as e:
            logger.error(f"Error extracting repair procedures: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_fault_ecu_relationships(self):
        """Extract fault-ECU relationships from RG_ECUFAULT_DOCIDS and fault code ECU references."""
        if not self._table_exists("XEP_FAULTCODES"):
            logger.info("XEP_FAULTCODES table not found, skipping fault-ECU relationship extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # Method 1: Via RG_ECUFAULT_DOCIDS (if it has ECU information)
                # First try to get fault-ECU relationships from fault codes directly
                try:
                    result = session.execute(
                        text("""
                            SELECT DISTINCT fc.ID as fault_id, fc.CODE, fc.ECUVARIANTID as ecu_id
                            FROM XEP_FAULTCODES fc
                            WHERE fc.CODE IS NOT NULL 
                              AND fc.ECUVARIANTID IS NOT NULL
                        """)
                    )
                    rows = result.fetchall()
                    
                    count = 0
                    for row in rows:
                        try:
                            # Access columns by index or by alias name (SQLAlchemy returns Row objects)
                            if hasattr(row, '_mapping'):
                                fault_code = str(row._mapping.get('CODE', '')) if row._mapping.get('CODE') else None
                                ecu_id = str(row._mapping.get('ecu_id', '')) if row._mapping.get('ecu_id') else None
                            elif hasattr(row, '_asdict'):
                                row_dict = row._asdict()
                                fault_code = str(row_dict.get('CODE', '')) if row_dict.get('CODE') else None
                                ecu_id = str(row_dict.get('ecu_id', '')) if row_dict.get('ecu_id') else None
                            else:
                                # Fallback: access by index (CODE is index 1, ecu_id is index 2)
                                fault_code = str(row[1]) if len(row) > 1 and row[1] else None
                                ecu_id = str(row[2]) if len(row) > 2 and row[2] else None
                            
                            if not fault_code or not ecu_id:
                                continue
                            
                            fault_node = f"fault_code:{fault_code}"
                            ecu_node = f"ecu:{ecu_id}"
                            
                            # Only add edge if both nodes exist
                            if fault_node in self.graph and ecu_node in self.graph:
                                self.graph.add_edge(
                                    fault_node,
                                    ecu_node,
                                    relationship="affects_ecu",
                                    weight=1.0
                                )
                                count += 1
                                self.stats['edges_added'] += 1
                        except Exception as e:
                            logger.warning(f"Error processing fault-ECU relationship: {e}")
                            self.stats['errors'] += 1
                            continue
                    
                    logger.info(f"Extracted {count} fault-ECU relationships")
                except Exception as e:
                    logger.warning(f"Could not extract fault-ECU relationships via ECUVARIANTID: {e}")
                    self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error extracting fault-ECU relationships: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_fault_diagnostic_relationships(self):
        """Extract fault-diagnostic relationships from XEP_REFDIAGOBJECTS table."""
        if not self._table_exists("XEP_REFDIAGOBJECTS"):
            logger.info("XEP_REFDIAGOBJECTS table not found, skipping fault-diagnostic relationship extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # XEP_REFDIAGOBJECTS.ID is the fault code ID, DIAGNOSISOBJECTCONTROLID links to diagnostic objects
                result = session.execute(
                    text("""
                        SELECT DISTINCT rdo.ID as fault_id, rdo.DIAGNOSISOBJECTCONTROLID as diag_control_id, rdo.PRIORITY
                        FROM XEP_REFDIAGOBJECTS rdo
                        INNER JOIN XEP_FAULTCODES fc ON rdo.ID = fc.ID
                        WHERE fc.CODE IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by alias or index
                        if hasattr(row, '_mapping'):
                            fault_id = str(row._mapping.get('fault_id', '')) if row._mapping.get('fault_id') else None
                            diag_control_id = str(row._mapping.get('diag_control_id', '')) if row._mapping.get('diag_control_id') else None
                            priority = float(row._mapping.get('PRIORITY', 1.0)) if row._mapping.get('PRIORITY') is not None else 1.0
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            fault_id = str(row_dict.get('fault_id', '')) if row_dict.get('fault_id') else None
                            diag_control_id = str(row_dict.get('diag_control_id', '')) if row_dict.get('diag_control_id') else None
                            priority = float(row_dict.get('PRIORITY', 1.0)) if row_dict.get('PRIORITY') is not None else 1.0
                        else:
                            # Fallback: access by index
                            fault_id = str(row[0]) if len(row) > 0 and row[0] else None
                            diag_control_id = str(row[1]) if len(row) > 1 and row[1] else None
                            priority = float(row[2]) if len(row) > 2 and row[2] is not None else 1.0
                        
                        if not fault_id or not diag_control_id:
                            continue
                        
                        # Get fault code from fault_id
                        fault_code_result = session.execute(
                            text("SELECT CODE FROM XEP_FAULTCODES WHERE ID = :fault_id LIMIT 1"),
                            {"fault_id": fault_id}
                        )
                        fault_code_row = fault_code_result.fetchone()
                        if not fault_code_row or not fault_code_row[0]:
                            continue
                        
                        fault_code = str(fault_code_row[0])
                        fault_node = f"fault_code:{fault_code}"
                        # Use DIAGNOSISOBJECTCONTROLID to find the diagnostic object
                        # We need to find the diagnostic object by CONTROLID
                        diag_obj_result = session.execute(
                            text("SELECT ID FROM XEP_DIAGNOSISOBJECTS WHERE CONTROLID = :control_id LIMIT 1"),
                            {"control_id": diag_control_id}
                        )
                        diag_obj_row = diag_obj_result.fetchone()
                        if not diag_obj_row or not diag_obj_row[0]:
                            continue
                        
                        diag_id = str(diag_obj_row[0])
                        diag_node = f"diagnostic:{diag_id}"
                        
                        # Only add edge if both nodes exist
                        if fault_node in self.graph and diag_node in self.graph:
                            # Weight based on priority (higher priority = higher weight)
                            weight = 1.0 / max(priority, 0.1) if priority > 0 else 1.0
                            
                            self.graph.add_edge(
                                fault_node,
                                diag_node,
                                relationship="has_diagnostic",
                                weight=weight,
                                priority=priority
                            )
                            count += 1
                            self.stats['edges_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing fault-diagnostic relationship: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} fault-diagnostic relationships")
        except Exception as e:
            logger.error(f"Error extracting fault-diagnostic relationships: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_fault_repair_relationships(self):
        """Extract fault-repair relationships from RG_ECUFAULT_DOCIDS table."""
        if not self._table_exists("RG_ECUFAULT_DOCIDS"):
            logger.info("RG_ECUFAULT_DOCIDS table not found, skipping fault-repair relationship extraction")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # RG_ECUFAULT_DOCIDS uses ECUFAULT_ID (not FAULTCODE_ID) and INFOOBJECTID (not DOCID)
                result = session.execute(
                    text("""
                        SELECT DISTINCT rg.ECUFAULT_ID as fault_id, rg.INFOOBJECTID as proc_id
                        FROM RG_ECUFAULT_DOCIDS rg
                        INNER JOIN XEP_FAULTCODES fc ON rg.ECUFAULT_ID = fc.ID
                        WHERE fc.CODE IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by alias or index
                        if hasattr(row, '_mapping'):
                            fault_id = str(row._mapping.get('fault_id', '')) if row._mapping.get('fault_id') else None
                            proc_id = str(row._mapping.get('proc_id', '')) if row._mapping.get('proc_id') else None
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            fault_id = str(row_dict.get('fault_id', '')) if row_dict.get('fault_id') else None
                            proc_id = str(row_dict.get('proc_id', '')) if row_dict.get('proc_id') else None
                        else:
                            # Fallback: access by index
                            fault_id = str(row[0]) if len(row) > 0 and row[0] else None
                            proc_id = str(row[1]) if len(row) > 1 and row[1] else None
                        
                        if not fault_id or not proc_id:
                            continue
                        
                        # Get fault code from fault_id
                        fault_code_result = session.execute(
                            text("SELECT CODE FROM XEP_FAULTCODES WHERE ID = :fault_id LIMIT 1"),
                            {"fault_id": fault_id}
                        )
                        fault_code_row = fault_code_result.fetchone()
                        if not fault_code_row or not fault_code_row[0]:
                            continue
                        
                        fault_code = str(fault_code_row[0])
                        fault_node = f"fault_code:{fault_code}"
                        proc_node = f"procedure:{proc_id}"
                        
                        # Only add edge if both nodes exist
                        if fault_node in self.graph and proc_node in self.graph:
                            self.graph.add_edge(
                                fault_node,
                                proc_node,
                                relationship="has_repair",
                                weight=1.0
                            )
                            count += 1
                            self.stats['edges_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing fault-repair relationship: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} fault-repair relationships")
        except Exception as e:
            logger.error(f"Error extracting fault-repair relationships: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _extract_diagnostic_tree_relationships(self):
        """Extract diagnostic tree parent-child relationships from XEP_REFDIAGNOSISTREE table."""
        if not self._table_exists("XEP_REFDIAGNOSISTREE"):
            logger.info("XEP_REFDIAGNOSISTREE table not found, skipping diagnostic tree relationships")
            return
        
        try:
            with self.ista_db.connection.session() as session:
                # XEP_REFDIAGNOSISTREE uses ID (parent) and DIAGNOSISOBJECTCONTROLID (child)
                result = session.execute(
                    text("""
                        SELECT DISTINCT rdt.ID as parent_control_id, rdt.DIAGNOSISOBJECTCONTROLID as child_control_id
                        FROM XEP_REFDIAGNOSISTREE rdt
                        WHERE rdt.ID IS NOT NULL AND rdt.DIAGNOSISOBJECTCONTROLID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        # Access columns by alias or index
                        if hasattr(row, '_mapping'):
                            parent_control_id = str(row._mapping.get('parent_control_id', '')) if row._mapping.get('parent_control_id') else None
                            child_control_id = str(row._mapping.get('child_control_id', '')) if row._mapping.get('child_control_id') else None
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                            parent_control_id = str(row_dict.get('parent_control_id', '')) if row_dict.get('parent_control_id') else None
                            child_control_id = str(row_dict.get('child_control_id', '')) if row_dict.get('child_control_id') else None
                        else:
                            # Fallback: access by index
                            parent_control_id = str(row[0]) if len(row) > 0 and row[0] else None
                            child_control_id = str(row[1]) if len(row) > 1 and row[1] else None
                        
                        if not parent_control_id or not child_control_id:
                            continue
                        
                        # Convert CONTROLID to diagnostic object ID
                        parent_obj_result = session.execute(
                            text("SELECT ID FROM XEP_DIAGNOSISOBJECTS WHERE CONTROLID = :control_id LIMIT 1"),
                            {"control_id": parent_control_id}
                        )
                        parent_obj_row = parent_obj_result.fetchone()
                        if not parent_obj_row or not parent_obj_row[0]:
                            continue
                        
                        child_obj_result = session.execute(
                            text("SELECT ID FROM XEP_DIAGNOSISOBJECTS WHERE CONTROLID = :control_id LIMIT 1"),
                            {"control_id": child_control_id}
                        )
                        child_obj_row = child_obj_result.fetchone()
                        if not child_obj_row or not child_obj_row[0]:
                            continue
                        
                        parent_id = str(parent_obj_row[0])
                        child_id = str(child_obj_row[0])
                        parent_node = f"diagnostic:{parent_id}"
                        child_node = f"diagnostic:{child_id}"
                        
                        # Only add edge if both nodes exist
                        if parent_node in self.graph and child_node in self.graph:
                            self.graph.add_edge(
                                parent_node,
                                child_node,
                                relationship="diagnostic_step",
                                weight=1.0
                            )
                            count += 1
                            self.stats['edges_added'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing diagnostic tree relationship: {e}")
                        self.stats['errors'] += 1
                        continue
                
                logger.info(f"Extracted {count} diagnostic tree relationships")
        except Exception as e:
            logger.warning(f"Error extracting diagnostic tree relationships: {e}")
            # Don't increment error count as this is optional
    
    def save(self, output_path: str | Path):
        """
        Save graph to GraphML file.
        
        Args:
            output_path: Path to output GraphML file
        """
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a copy of the graph for saving (GraphML doesn't support dict/list values)
        graph_to_save = nx.MultiDiGraph()
        
        # Copy nodes, converting complex types to strings
        for node_id, data in self.graph.nodes(data=True):
            node_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    node_data[key] = str(value)
                else:
                    node_data[key] = value
            graph_to_save.add_node(node_id, **node_data)
        
        # Copy edges, converting complex types to strings
        for source, target, data in self.graph.edges(data=True):
            edge_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    edge_data[key] = str(value)
                else:
                    edge_data[key] = value
            graph_to_save.add_edge(source, target, **edge_data)
        
        # Copy graph-level attributes, converting complex types to strings
        for key, value in self.graph.graph.items():
            if isinstance(value, (dict, list)):
                graph_to_save.graph[key] = str(value)
            else:
                graph_to_save.graph[key] = value
        
        # Write graph to GraphML format
        nx.write_graphml(graph_to_save, str(output_path))
        
        logger.info(f"Saved knowledge graph to {output_path}")
        logger.info(
            f"Graph contains {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

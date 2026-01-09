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
                result = session.execute(
                    text("""
                        SELECT DISTINCT fc.ID, fc.CODE, fl.TITLE_ENGB, fl.LABEL
                        FROM XEP_FAULTCODES fc
                        LEFT JOIN XEP_FAULTLABELS fl ON fc.ID = fl.FAULTCODE_ID
                        WHERE fc.CODE IS NOT NULL AND fc.CODE != ''
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        fault_id = str(row.ID) if row.ID else None
                        code = str(row.CODE) if row.CODE else None
                        
                        if not fault_id or not code:
                            continue
                        
                        node_id = f"fault_code:{code}"
                        
                        # Skip if node already exists (incremental mode)
                        if node_id in self.graph:
                            continue
                        
                        # Get attributes
                        title_engb = str(row.TITLE_ENGB) if row.TITLE_ENGB else None
                        label = str(row.LABEL) if row.LABEL else None
                        
                        # Add node with attributes
                        self.graph.add_node(
                            node_id,
                            node_type="fault_code",
                            id=fault_id,
                            code=code,
                            title_engb=title_engb or label or "",
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
                        SELECT DISTINCT ev.ID, ev.NAME, ev.TYPEKEY
                        FROM XEP_ECUVARIANTS ev
                        WHERE ev.ID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        ecu_id = str(row.ID) if row.ID else None
                        if not ecu_id:
                            continue
                        
                        node_id = f"ecu:{ecu_id}"
                        
                        # Skip if node already exists
                        if node_id in self.graph:
                            continue
                        
                        name = str(row.NAME) if row.NAME else ""
                        type_key = str(row.TYPEKEY) if row.TYPEKEY else None
                        
                        self.graph.add_node(
                            node_id,
                            node_type="ecu",
                            id=ecu_id,
                            name=name,
                            type_key=type_key or "",
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
                        SELECT DISTINCT do.ID, do.TITLE
                        FROM XEP_DIAGNOSISOBJECTS do
                        WHERE do.ID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        diag_id = str(row.ID) if row.ID else None
                        if not diag_id:
                            continue
                        
                        node_id = f"diagnostic:{diag_id}"
                        
                        # Skip if node already exists
                        if node_id in self.graph:
                            continue
                        
                        title = str(row.TITLE) if row.TITLE else ""
                        
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
                            fault_code = str(row.CODE) if row.CODE else None
                            ecu_id = str(row.ECUVARIANTID) if row.ECUVARIANTID else None
                            
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
                result = session.execute(
                    text("""
                        SELECT DISTINCT rdo.FAULTCODE_ID, rdo.DIAGNOSISOBJECTID, rdo.PRIORITY
                        FROM XEP_REFDIAGOBJECTS rdo
                        INNER JOIN XEP_FAULTCODES fc ON rdo.FAULTCODE_ID = fc.ID
                        WHERE fc.CODE IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        fault_id = str(row.FAULTCODE_ID) if row.FAULTCODE_ID else None
                        diag_id = str(row.DIAGNOSISOBJECTID) if row.DIAGNOSISOBJECTID else None
                        priority = float(row.PRIORITY) if row.PRIORITY is not None else 1.0
                        
                        if not fault_id or not diag_id:
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
                result = session.execute(
                    text("""
                        SELECT DISTINCT rg.FAULTCODE_ID, rg.DOCID
                        FROM RG_ECUFAULT_DOCIDS rg
                        INNER JOIN XEP_FAULTCODES fc ON rg.FAULTCODE_ID = fc.ID
                        WHERE fc.CODE IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        fault_id = str(row.FAULTCODE_ID) if row.FAULTCODE_ID else None
                        doc_id = str(row.DOCID) if row.DOCID else None
                        
                        if not fault_id or not doc_id:
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
                        proc_node = f"procedure:{doc_id}"
                        
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
                
                result = session.execute(
                    text("""
                        SELECT DISTINCT rdt.PARENTID, rdt.CHILDID
                        FROM XEP_REFDIAGNOSISTREE rdt
                        WHERE rdt.PARENTID IS NOT NULL AND rdt.CHILDID IS NOT NULL
                    """)
                )
                rows = result.fetchall()
                
                count = 0
                for row in rows:
                    try:
                        parent_id = str(row.PARENTID) if row.PARENTID else None
                        child_id = str(row.CHILDID) if row.CHILDID else None
                        
                        if not parent_id or not child_id:
                            continue
                        
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

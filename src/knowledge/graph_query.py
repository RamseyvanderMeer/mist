"""
Knowledge graph query interface for path finding and relationship reasoning.
"""
import networkx as nx

from src.database.fault_code_mapping import get_lookup_variants
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphQuery:
    """
    Query interface for knowledge graph path finding and scoring.
    
    Supports finding paths from fault codes to repair procedures, diagnostic objects,
    and ECUs, with weighted path scoring based on edge relationships.
    """
    
    def __init__(self, graph_path: str | Path):
        """
        Initialize graph query interface.
        
        Args:
            graph_path: Path to GraphML file containing the knowledge graph
        
        Raises:
            FileNotFoundError: If graph file doesn't exist (logs warning, uses empty graph)
        """
        self.graph_path = Path(graph_path)
        if self.graph_path.exists():
            try:
                self.graph = nx.read_graphml(str(self.graph_path))
                # Ensure we have a MultiDiGraph
                if not isinstance(self.graph, nx.MultiDiGraph):
                    logger.warning(
                        f"Graph loaded is not MultiDiGraph, converting from {type(self.graph)}"
                    )
                    self.graph = nx.MultiDiGraph(self.graph)
                logger.info(
                    f"Loaded knowledge graph from {graph_path} "
                    f"({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)"
                )
            except Exception as e:
                logger.error(f"Error loading graph from {graph_path}: {e}", exc_info=True)
                self.graph = nx.MultiDiGraph()
                logger.warning("Using empty graph due to load error")
        else:
            self.graph = nx.MultiDiGraph()
            logger.warning(f"Knowledge graph not found at {graph_path}, using empty graph")
    
    def find_paths(
        self,
        source_node: str,
        target_type: str,
        max_length: int = 3,
        target_node: Optional[str] = None
    ) -> List[List[str]]:
        """
        Find paths from source node to nodes of target type.
        
        Uses weighted shortest path algorithm to find optimal paths based on edge weights.
        For MultiDiGraph, considers all edges between nodes and selects the best path.
        
        Args:
            source_node: Source node ID (e.g., "fault_code:12345")
            target_type: Target node type (e.g., "procedure", "diagnostic", "ecu")
            max_length: Maximum path length (number of edges)
            target_node: Optional specific target node ID. If provided, only finds paths to this node.
        
        Returns:
            List of paths (each path is a list of node IDs). Paths are sorted by length.
            Returns empty list if source node doesn't exist or no paths found.
        """
        if source_node not in self.graph:
            logger.debug(f"Source node {source_node} not found in graph")
            return []
        
        # Find target nodes of specified type
        if target_node:
            # Verify target node exists and matches type
            if target_node not in self.graph:
                logger.debug(f"Target node {target_node} not found in graph")
                return []
            node_data = self.graph.nodes[target_node]
            if node_data.get("node_type") != target_type:
                logger.debug(
                    f"Target node {target_node} has type {node_data.get('node_type')}, "
                    f"expected {target_type}"
                )
                return []
            target_nodes = [target_node]
        else:
            target_nodes = [
                node for node, data in self.graph.nodes(data=True)
                if data.get("node_type") == target_type
            ]
        
        if not target_nodes:
            logger.debug(f"No target nodes found with type {target_type}")
            return []
        
        paths = []
        for target in target_nodes:
            try:
                # Use weighted shortest path if edges have weight attribute
                # Check if any edges have weight attribute
                has_weights = False
                for u, v, data in self.graph.edges(data=True):
                    if "weight" in data:
                        has_weights = True
                        break
                
                if has_weights:
                    # Use weighted shortest path
                    try:
                        path = nx.shortest_path(
                            self.graph,
                            source_node,
                            target,
                            weight="weight",
                            method="dijkstra"
                        )
                    except nx.NetworkXNoPath:
                        continue
                else:
                    # Fall back to unweighted shortest path
                    path = nx.shortest_path(self.graph, source_node, target)
                
                # Check path length constraint (max_length is number of edges)
                if len(path) - 1 <= max_length:
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                logger.warning(f"Error finding path from {source_node} to {target}: {e}")
                continue
        
        # Sort paths by length (shortest first)
        paths.sort(key=len)
        
        # Return top paths (limit to reasonable number)
        return paths[:20]
    
    def score_path(self, path: List[str]) -> float:
        """
        Score a single path based on edge weights.
        
        Calculates the sum of edge weights along the path, normalized by path length.
        For MultiDiGraph, selects the minimum weight edge between nodes (most conservative).
        
        Args:
            path: List of node IDs forming a path
        
        Returns:
            Path score (higher is better). Returns 0.0 for invalid paths.
        """
        if len(path) < 2:
            return 0.0
        
        total_weight = 0.0
        edges_found = 0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            edges = self.graph.get_edge_data(source, target)
            
            if edges:
                # For MultiDiGraph, get all edge weights and use minimum (most conservative)
                weights = []
                for edge_data in edges.values():
                    weight = edge_data.get("weight")
                    if weight is not None:
                        try:
                            weights.append(float(weight))
                        except (ValueError, TypeError):
                            pass
                
                if weights:
                    total_weight += min(weights)
                    edges_found += 1
                else:
                    # No weight attribute, use default weight of 1.0
                    total_weight += 1.0
                    edges_found += 1
            else:
                # Missing edge in path - invalid path
                logger.debug(f"Missing edge between {source} and {target} in path")
                return 0.0
        
        if edges_found == 0:
            return 0.0
        
        # Normalize by path length (number of edges)
        # Higher weight sum with shorter path = better score
        return total_weight / max(len(path) - 1, 1)
    
    def score_paths(self, paths: List[List[str]]) -> List[float]:
        """
        Score multiple paths based on edge weights.
        
        Args:
            paths: List of paths (each path is a list of node IDs)
        
        Returns:
            List of scores corresponding to each path. Empty list if input is empty.
        """
        if not paths:
            return []
        
        return [self.score_path(path) for path in paths]
    
    def get_node_by_code(self, fault_code: str) -> Optional[str]:
        """
        Find fault code node by code string.
        
        Args:
            fault_code: Fault code string (e.g., "12345")
        
        Returns:
            Node ID if found (e.g., "fault_code:12345"), None otherwise
        """
        node_id = f"fault_code:{fault_code}"
        if node_id in self.graph:
            return node_id
        
        # Fallback: search by code attribute
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "fault_code" and data.get("code") == fault_code:
                return node
        
        return None
    
    def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes connected to a given node.
        
        Args:
            node_id: Node ID to get neighbors for
            relationship_type: Optional filter by relationship type (e.g., "has_repair")
        
        Returns:
            List of dicts with neighbor information:
            - node_id: Neighbor node ID
            - node_type: Type of neighbor node
            - relationship: Relationship type
            - weight: Edge weight
        """
        if node_id not in self.graph:
            return []
        
        neighbors = []
        for successor in self.graph.successors(node_id):
            edges = self.graph.get_edge_data(node_id, successor)
            if not edges:
                continue
            
            for edge_key, edge_data in edges.items():
                rel_type = edge_data.get("relationship", "")
                
                # Filter by relationship type if specified
                if relationship_type and rel_type != relationship_type:
                    continue
                
                neighbor_data = self.graph.nodes[successor]
                neighbors.append({
                    "node_id": successor,
                    "node_type": neighbor_data.get("node_type", ""),
                    "relationship": rel_type,
                    "weight": edge_data.get("weight", 1.0),
                    "node_data": dict(neighbor_data)  # Include all node attributes
                })
        
        return neighbors
    
    def get_procedures_for_fault(self, fault_code: str, max_length: int = 3) -> List[Dict[str, Any]]:
        """
        Get repair procedures for a fault code with path scores.
        
        Finds paths from fault code node to procedure nodes and scores them.
        
        Args:
            fault_code: Fault code string (e.g., "12345")
            max_length: Maximum path length to consider
        
        Returns:
            List of dicts with procedure info and path scores, sorted by score (highest first).
            Each dict contains:
            - procedure_id: Procedure ID
            - procedure_title: Procedure title/name
            - path_score: Score of the path from fault to procedure
            - path: List of node IDs forming the path
        """
        # Find fault node - try variants (P-code -> BMW hex) since graph uses ISTA codes
        fault_node = None
        for variant in get_lookup_variants(fault_code):
            fault_node = self.get_node_by_code(variant)
            if fault_node:
                break
        if not fault_node:
            logger.debug(f"Fault code {fault_code} not found in graph (tried: {get_lookup_variants(fault_code)})")
            return []
        
        # Find paths to repair procedures (using correct target type)
        paths = self.find_paths(fault_node, "procedure", max_length=max_length)
        
        if not paths:
            logger.debug(f"No paths found from {fault_node} to procedures")
            return []
        
        # Score all paths
        path_scores = self.score_paths(paths)
        
        procedures = []
        for path, score in zip(paths, path_scores):
            procedure_node = path[-1]
            
            if procedure_node not in self.graph:
                continue
            
            procedure_data = self.graph.nodes[procedure_node]
            
            procedures.append({
                "procedure_id": procedure_data.get("id", ""),
                "procedure_title": procedure_data.get("title_engb", "") or procedure_data.get("name", ""),
                "path_score": score,
                "path": path,
                "path_length": len(path) - 1
            })
        
        # Sort by score (highest first), then by path length (shortest first)
        return sorted(
            procedures,
            key=lambda x: (x["path_score"], -x["path_length"]),
            reverse=True
        )

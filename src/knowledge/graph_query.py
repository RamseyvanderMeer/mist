"""
Knowledge graph query interface for path finding and relationship reasoning.
"""
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphQuery:
    """
    Query interface for knowledge graph path finding and scoring.
    """
    
    def __init__(self, graph_path: str):
        """
        Initialize graph query interface.
        
        Args:
            graph_path: Path to GraphML file
        """
        self.graph_path = Path(graph_path)
        if self.graph_path.exists():
            self.graph = nx.read_graphml(graph_path)
            logger.info(f"Loaded knowledge graph from {graph_path}")
        else:
            self.graph = nx.MultiDiGraph()
            logger.warning(f"Knowledge graph not found at {graph_path}, using empty graph")
    
    def find_paths(self, source_node: str, target_type: str, max_length: int = 3) -> List[List[str]]:
        """
        Find paths from source node to nodes of target type.
        
        Args:
            source_node: Source node ID
            target_type: Target node type (e.g., "repair_procedure")
            max_length: Maximum path length
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source_node not in self.graph:
            return []
        
        # Find target nodes of specified type
        target_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("type") == target_type
        ]
        
        paths = []
        for target in target_nodes[:10]:  # Limit to top 10 targets
            try:
                path = nx.shortest_path(self.graph, source_node, target)
                if len(path) <= max_length + 1:
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
        
        return paths[:10]  # Return top 10 paths
    
    def score_path(self, path: List[str]) -> float:
        """
        Score path based on edge weights.
        
        Args:
            path: List of node IDs forming a path
        
        Returns:
            Path score (0.0 to 1.0)
        """
        if len(path) < 2:
            return 0.0
        
        total_weight = 0.0
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            edges = self.graph.get_edge_data(source, target)
            if edges:
                # Get minimum weight (for MultiDiGraph)
                weights = [e.get("weight", 1.0) for e in edges.values()]
                total_weight += min(weights) if weights else 1.0
        
        # Normalize by path length
        return total_weight / max(len(path) - 1, 1)
    
    def get_procedures_for_fault(self, fault_code: str) -> List[Dict]:
        """
        Get repair procedures for a fault code.
        
        Args:
            fault_code: Fault code string
        
        Returns:
            List of dicts with procedure info and path scores
        """
        # Find fault node
        fault_node = f"fault_{fault_code}"
        if fault_node not in self.graph:
            return []
        
        # Find paths to repair procedures
        paths = self.find_paths(fault_node, "repair_procedure", max_length=3)
        
        procedures = []
        for path in paths:
            procedure_node = path[-1]
            procedure_data = self.graph.nodes[procedure_node]
            score = self.score_path(path)
            
            procedures.append({
                "procedure_id": procedure_data.get("id", ""),
                "procedure_name": procedure_data.get("name", ""),
                "path_score": score,
                "path": path
            })
        
        return sorted(procedures, key=lambda x: x["path_score"], reverse=True)

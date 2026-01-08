#!/usr/bin/env python3
"""
Build knowledge graph from BMW diagnostic database.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge.graph_builder import KnowledgeGraphBuilder
from paths import get_paths
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """Build knowledge graph"""
    paths = get_paths()
    
    # Get primary database
    db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
    if not db_path.exists():
        db_path = paths.get_database_path("DiagDocDb_Decrypted.sqlite")
    
    if not db_path.exists():
        print(f"Error: Database not found. Checked: {db_path}")
        return
    
    print(f"Building knowledge graph from: {db_path}")
    
    builder = KnowledgeGraphBuilder(str(db_path))
    graph = builder.build()
    
    # Save graph
    output_path = paths.knowledge_graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(str(output_path))
    
    print(f"Knowledge graph saved to: {output_path}")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

if __name__ == "__main__":
    main()

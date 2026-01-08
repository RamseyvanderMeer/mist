#!/usr/bin/env python3
"""
Build knowledge graph from BMW diagnostic database.

Usage:
    python build_knowledge_graph.py [--incremental] [--output PATH]
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge.graph_builder import KnowledgeGraphBuilder
from database.ista_db import IstaDatabase
from paths import get_paths
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Build knowledge graph from BMW diagnostic database."""
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from BMW ISTA diagnostic database"
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Merge with existing graph instead of rebuilding from scratch'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for GraphML file (default: from paths.knowledge_graph)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to BMW diagnostic database (default: auto-detect)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    paths = get_paths()
    
    # Get database path
    if args.db_path:
        db_path = Path(args.db_path)
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            sys.exit(1)
    else:
        # Try to find database
        db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
        if not db_path.exists():
            db_path = paths.get_database_path("DiagDocDb_Decrypted.sqlite")
        
        if not db_path.exists():
            logger.error(f"Database not found. Checked: {db_path}")
            logger.error("Please specify database path with --db-path")
            sys.exit(1)
    
    logger.info(f"Building knowledge graph from: {db_path}")
    logger.info(f"Incremental mode: {args.incremental}")
    
    try:
        # Initialize ISTA database wrapper
        ista_db = IstaDatabase(db_path)
        
        # Test connection
        if not ista_db.test_connection():
            logger.error("Failed to connect to database")
            sys.exit(1)
        
        # Create builder
        builder = KnowledgeGraphBuilder(ista_db=ista_db, incremental=args.incremental)
        
        # Build graph
        logger.info("Starting graph construction...")
        graph = builder.build()
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = paths.knowledge_graph
        
        # Save graph
        logger.info(f"Saving knowledge graph to: {output_path}")
        builder.save(output_path)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Knowledge Graph Build Complete")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_path}")
        logger.info(f"Total nodes: {graph.number_of_nodes()}")
        logger.info(f"Total edges: {graph.number_of_edges()}")
        logger.info(f"Nodes added: {builder.stats['nodes_added']}")
        logger.info(f"Edges added: {builder.stats['edges_added']}")
        logger.info(f"Errors encountered: {builder.stats['errors']}")
        
        # Print node type breakdown
        node_types = {}
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        logger.info("\nNode type breakdown:")
        for node_type, count in sorted(node_types.items()):
            logger.info(f"  {node_type}: {count}")
        
        # Print edge type breakdown
        edge_types = {}
        for source, target, data in graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
        
        logger.info("\nEdge type breakdown:")
        for rel_type, count in sorted(edge_types.items()):
            logger.info(f"  {rel_type}: {count}")
        
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

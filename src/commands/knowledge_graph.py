"""Build knowledge graph from BMW ISTA diagnostic database."""
import logging
from pathlib import Path

import networkx as nx

from src.knowledge.graph_builder import KnowledgeGraphBuilder
from src.database.ista_db import IstaDatabase
from src.paths import get_paths

logger = logging.getLogger(__name__)


def run(
    incremental: bool = False,
    output: str | None = None,
    db_path: str | None = None,
    verbose: bool = False,
) -> int:
    """Build knowledge graph. Returns 0 on success, 1 on failure."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    paths = get_paths()

    if db_path:
        db_path_p = Path(db_path)
        if not db_path_p.exists():
            logger.error("Database not found: %s", db_path_p)
            return 1
    else:
        db_path_p = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
        if not db_path_p.exists():
            db_path_p = paths.get_database_path("DiagDocDb_Decrypted.sqlite")
        if not db_path_p.exists():
            logger.error("Database not found. Checked: %s", db_path_p)
            logger.error("Please specify database path with --db-path")
            return 1

    output_path = Path(output) if output else paths.knowledge_graph

    try:
        logger.info("Building knowledge graph from: %s", db_path_p)
        logger.info("Incremental mode: %s", incremental)

        ista_db = IstaDatabase(db_path_p)
        if not ista_db.test_connection():
            logger.error("Failed to connect to database")
            return 1

        builder = KnowledgeGraphBuilder(ista_db=ista_db, incremental=incremental)

        if incremental and output_path.exists():
            try:
                logger.info("Loading existing knowledge graph from: %s", output_path)
                loaded = nx.read_graphml(str(output_path))
                builder.graph = nx.MultiDiGraph(loaded)
                skip_attrs = {"node_default", "edge_default", "defaultedgedefault", "defaultnodedefault"}
                for attr in skip_attrs:
                    if attr in builder.graph.graph:
                        del builder.graph.graph[attr]
                logger.info(
                    "Loaded graph with %d nodes and %d edges",
                    builder.graph.number_of_nodes(),
                    builder.graph.number_of_edges(),
                )
            except Exception as e:
                logger.warning("Failed to load existing graph (%s): %s. Continuing with empty graph.", output_path, e)

        logger.info("Starting graph construction...")
        graph = builder.build()

        logger.info("Saving knowledge graph to: %s", output_path)
        builder.save(output_path)

        logger.info("=" * 60)
        logger.info("Knowledge Graph Build Complete")
        logger.info("=" * 60)
        logger.info("Output file: %s", output_path)
        logger.info("Total nodes: %d", graph.number_of_nodes())
        logger.info("Total edges: %d", graph.number_of_edges())
        logger.info("Nodes added: %d", builder.stats["nodes_added"])
        logger.info("Edges added: %d", builder.stats["edges_added"])
        logger.info("Errors encountered: %d", builder.stats["errors"])

        node_types = {}
        for _nid, data in graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1
        logger.info("Node type breakdown:")
        for nt, count in sorted(node_types.items()):
            logger.info("  %s: %d", nt, count)

        edge_types = {}
        for _s, _t, data in graph.edges(data=True):
            rt = data.get("relationship", "unknown")
            edge_types[rt] = edge_types.get(rt, 0) + 1
        logger.info("Edge type breakdown:")
        for rt, count in sorted(edge_types.items()):
            logger.info("  %s: %d", rt, count)

        return 0
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        return 1
    except Exception as e:
        logger.error("Error building knowledge graph: %s", e, exc_info=True)
        return 1

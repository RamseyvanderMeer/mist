#!/usr/bin/env python3
"""
Migrate Qdrant collection from local storage to cloud Qdrant.

This script reads all vectors from a local Qdrant collection and uploads them
to a cloud Qdrant instance. Useful for migrating data after local indexing.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest, ScrollResult
from retrieval.vector_store import VectorStore
from paths import get_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_collection(
    local_path: str,
    cloud_url: str,
    cloud_api_key: str,
    collection_name: str,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Migrate collection from local Qdrant to cloud Qdrant.
    
    Args:
        local_path: Path to local Qdrant storage
        cloud_url: Cloud Qdrant URL
        cloud_api_key: Cloud Qdrant API key
        collection_name: Collection name to migrate
        batch_size: Batch size for uploading
    
    Returns:
        Dictionary with migration statistics
    """
    logger.info(f"Connecting to local Qdrant at: {local_path}")
    local_client = QdrantClient(path=local_path)
    
    logger.info(f"Connecting to cloud Qdrant at: {cloud_url}")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)
    
    # Check if collection exists locally
    local_collections = local_client.get_collections().collections
    local_collection_names = [c.name for c in local_collections]
    
    if collection_name not in local_collection_names:
        raise ValueError(f"Collection '{collection_name}' not found in local Qdrant")
    
    logger.info(f"Found local collection: {collection_name}")
    
    # Get collection info from local
    local_info = local_client.get_collection(collection_name)
    vector_size = local_info.config.params.vectors.size
    distance = local_info.config.params.vectors.distance
    
    logger.info(f"Collection config: size={vector_size}, distance={distance}")
    logger.info(f"Total points in local collection: {local_info.points_count}")
    
    # Check if collection exists in cloud, create if not
    cloud_collections = cloud_client.get_collections().collections
    cloud_collection_names = [c.name for c in cloud_collections]
    
    if collection_name not in cloud_collection_names:
        logger.info(f"Creating collection '{collection_name}' in cloud...")
        from qdrant_client.models import VectorParams, Distance as QdrantDistance
        
        distance_map = {
            "Cosine": QdrantDistance.COSINE,
            "Euclid": QdrantDistance.EUCLID,
            "Dot": QdrantDistance.DOT,
        }
        
        cloud_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(str(distance), QdrantDistance.COSINE)
            )
        )
        logger.info(f"Created collection '{collection_name}' in cloud")
    else:
        logger.info(f"Collection '{collection_name}' already exists in cloud")
        cloud_info = cloud_client.get_collection(collection_name)
        logger.info(f"Current points in cloud collection: {cloud_info.points_count}")
    
    # Scroll through all points in local collection
    logger.info("Starting migration...")
    migrated_count = 0
    batch_points = []
    
    # Use scroll to get all points
    offset = None
    while True:
        scroll_result: ScrollResult = local_client.scroll(
            collection_name=collection_name,
            scroll_request=ScrollRequest(
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
        )
        
        points = scroll_result.points
        if not points:
            break
        
        batch_points.extend(points)
        
        # Upload batch when full
        if len(batch_points) >= batch_size:
            cloud_client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            migrated_count += len(batch_points)
            logger.info(f"Migrated {migrated_count}/{local_info.points_count} points...")
            batch_points = []
        
        # Check if we've reached the end
        if len(points) < batch_size:
            break
        
        offset = scroll_result.next_page_offset
        if offset is None:
            break
    
    # Upload remaining points
    if batch_points:
        cloud_client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        migrated_count += len(batch_points)
    
    logger.info(f"Migration complete! Migrated {migrated_count} points")
    
    # Verify
    final_cloud_info = cloud_client.get_collection(collection_name)
    logger.info(f"Final cloud collection size: {final_cloud_info.points_count} points")
    
    return {
        "local_points": local_info.points_count,
        "migrated_points": migrated_count,
        "cloud_points": final_cloud_info.points_count,
        "success": final_cloud_info.points_count == local_info.points_count
    }


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection from local to cloud"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="./data/vector_store",
        help="Path to local Qdrant storage (default: ./data/vector_store)"
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        required=True,
        help="Cloud Qdrant URL (e.g., https://your-cluster.qdrant.io)"
    )
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="Cloud Qdrant API key (or set QDRANT_API_KEY env var)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Collection name (default: from retrieval_config.yaml)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to retrieval_config.yaml (default: from paths.py)"
    )
    
    args = parser.parse_args()
    
    # Load collection name from config if not provided
    collection_name = args.collection_name
    if not collection_name:
        if args.config:
            config_path = Path(args.config)
        else:
            paths = get_paths()
            config_path = paths.retrieval_config
        
        with open(config_path, 'r') as f:
            retrieval_config = yaml.safe_load(f)
        
        collection_name = retrieval_config.get("vector_store", {}).get("collection_name", "repair_guides_enhanced")
        logger.info(f"Using collection name from config: {collection_name}")
    
    # Get API key from env or arg
    import os
    cloud_api_key = args.cloud_api_key or os.getenv("QDRANT_API_KEY")
    if not cloud_api_key:
        logger.warning("No API key provided. Some cloud instances may not require it.")
    
    try:
        stats = migrate_collection(
            local_path=args.local_path,
            cloud_url=args.cloud_url,
            cloud_api_key=cloud_api_key,
            collection_name=collection_name,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Local points: {stats['local_points']}")
        print(f"Migrated points: {stats['migrated_points']}")
        print(f"Cloud points: {stats['cloud_points']}")
        print(f"Success: {stats['success']}")
        print("="*60)
        
        if not stats['success']:
            logger.warning("Point counts don't match! Please verify the migration.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

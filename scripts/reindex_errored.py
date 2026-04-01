#!/usr/bin/env python3
"""
Re-index only the procedures that previously errored out.

Usage:
    python reindex_errored.py --batch-size 100
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Iterator, Dict, Any, Set
import logging
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.qwen3_encoder import Qwen3Encoder
from src.retrieval.chroma_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_errored_ids() -> Set[str]:
    """Get list of procedure IDs that failed to index."""
    checkpoint_file = Path(__file__).parent.parent / "data" / "qwen3_indexing_checkpoint.json"
    
    # Load indexed IDs from checkpoint
    indexed_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            indexed_ids = set(data.get('indexed_ids', []))
    
    # Get all procedure IDs from database
    import sqlite3
    db_path = Path(__file__).parent.parent / "data" / "databases" / "DiagDocDb_DECRYPTED.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT ID FROM XEP_INFOOBJECTS')
    all_ids = set(str(r[0]) for r in cursor.fetchall())
    conn.close()
    
    # Return IDs that haven't been indexed
    errored = all_ids - indexed_ids
    logger.info(f"Found {len(errored)} errored/missing procedures out of {len(all_ids)} total")
    logger.info(f"Already indexed: {len(indexed_ids)}")
    
    return errored


def get_repair_guides_for_ids(proc_ids: Set[str], batch_size: int = 100) -> Iterator[Dict[str, Any]]:
    """
    Fetch repair guides for specific procedure IDs.
    
    Yields dicts with: id, title, text, procedure_id, etc.
    """
    from src.database.ista_db import IstaDatabase
    from src.database.xml_content import XmlContentFetcher
    
    db_path = Path(__file__).parent.parent / "data" / "databases" / "DiagDocDb_DECRYPTED.sqlite"
    ista_db = IstaDatabase(db_path=db_path)
    xml_fetcher = XmlContentFetcher()
    
    success_count = 0
    error_count = 0
    
    # Sort for consistent ordering
    sorted_ids = sorted(proc_ids)
    logger.info(f"Processing {len(sorted_ids)} errored procedures...")
    
    for proc_id in sorted_ids:
        try:
            # Get procedure info
            info = ista_db.get_info_object(proc_id)
            if not info:
                error_count += 1
                continue
            
            title = info.get('TITLE_ENGB', '') or info.get('title', '')
            
            # Get XML content
            xml_content = xml_fetcher.get_content(proc_id, title)
            if not xml_content:
                error_count += 1
                # Log why it failed
                if error_count <= 5:
                    logger.info(f"No XML content for {proc_id}: {title[:50]}...")
                continue
            
            success_count += 1
            if success_count % 100 == 0:
                logger.info(f"Processed {success_count} documents successfully ({error_count} still errored)")
            
            # Get fault codes
            fault_codes = ista_db.get_fault_codes_for_procedure(proc_id)
            
            yield {
                "id": str(proc_id),
                "title": title,
                "text": xml_content,
                "procedure_id": str(proc_id),
                "procedure_name": title,
                "ecu_category": info.get('ecu_category', ''),
                "fault_codes": fault_codes
            }
            
        except Exception as e:
            logger.warning(f"Error processing {proc_id}: {e}")
            error_count += 1
            continue
    
    logger.info(f"Finished: {success_count} successful, {error_count} still errored")


def load_checkpoint() -> set:
    """Load already indexed procedure IDs from checkpoint file."""
    checkpoint_file = Path(__file__).parent.parent / "data" / "qwen3_indexing_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data.get('indexed_ids', []))
    return set()


def save_checkpoint(indexed_ids: set):
    """Save checkpoint of indexed procedure IDs."""
    checkpoint_file = Path(__file__).parent.parent / "data" / "qwen3_indexing_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'indexed_ids': list(indexed_ids),
            'count': len(indexed_ids),
            'last_updated': datetime.now().isoformat()
        }, f)
    logger.info(f"Checkpoint saved: {len(indexed_ids)} total indexed")


def reindex_errored(
    encoder: Qwen3Encoder,
    vector_store: ChromaVectorStore,
    batch_size: int = 100,
    max_docs: int = None
):
    """
    Re-index only the procedures that previously errored out.
    """
    collection = vector_store.client.get_collection("repair_guides_qwen3")
    logger.info(f"Using collection: repair_guides_qwen3")
    
    # Get IDs that need re-indexing
    errored_ids = get_errored_ids()
    if not errored_ids:
        logger.info("No errored procedures to re-index!")
        return
    
    # Load checkpoint to resume if needed
    already_indexed = load_checkpoint()
    logger.info(f"Resuming from checkpoint: {len(already_indexed)} already indexed")
    
    # Filter to only errored IDs not yet indexed
    ids_to_index = errored_ids - already_indexed
    logger.info(f"Will attempt to index {len(ids_to_index)} procedures")
    
    if max_docs:
        ids_to_index = set(list(ids_to_index)[:max_docs])
        logger.info(f"Limited to first {max_docs} procedures for testing")
    
    # Fetch and index documents
    total_indexed = len(already_indexed)
    batch_ids = []
    batch_texts = []
    batch_metadatas = []
    
    for doc in tqdm(get_repair_guides_for_ids(ids_to_index, batch_size), 
                    desc="Indexing errored", 
                    total=len(ids_to_index)):
        
        # Prepare text for embedding
        text = f"{doc['title']}\n{doc['text']}" if doc['title'] else doc['text']
        text = text[:8000]  # Truncate if too long
        
        batch_ids.append(doc['id'])
        batch_texts.append(text)
        
        # Build metadata (using optimized version if available)
        try:
            from metadata_utils import build_metadata
            meta = build_metadata(doc, text)
        except ImportError:
            # Fallback to simple metadata
            meta = {
                "title": doc['title'][:200] if doc['title'] else "",
                "procedure_id": doc['procedure_id'] or "",
                "text_preview": text[:500],
                "fault_codes": doc.get('fault_codes', ['P0000'])[:10]
            }
        
        batch_metadatas.append(meta)
        
        # Process batch
        if len(batch_ids) >= batch_size:
            try:
                embeddings = encoder.encode(batch_texts, is_query=False)
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    metadatas=batch_metadatas
                )
                
                total_indexed += len(batch_ids)
                already_indexed.update(batch_ids)
                
                if total_indexed % 500 == 0:
                    logger.info(f"Indexed {total_indexed} documents total...")
                    save_checkpoint(already_indexed)
                
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
            
            batch_ids = []
            batch_texts = []
            batch_metadatas = []
    
    # Process remaining batch
    if batch_ids:
        try:
            embeddings = encoder.encode(batch_texts, is_query=False)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                metadatas=batch_metadatas
            )
            total_indexed += len(batch_ids)
            already_indexed.update(batch_ids)
        except Exception as e:
            logger.error(f"Error indexing final batch: {e}")
    
    logger.info(f"Finished indexing {total_indexed} documents total")
    save_checkpoint(already_indexed)
    
    # Verify
    count = collection.count()
    logger.info(f"Collection now has {count} documents")


def main():
    parser = argparse.ArgumentParser(description="Re-index errored procedures")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for indexing")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to index (for testing)")
    parser.add_argument("--output-dim", type=int, default=4096, help="Output dimensions")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("NEBIUS_API_KEY"):
        logger.error("NEBIUS_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize encoder
    logger.info(f"Initializing Qwen3 encoder with {args.output_dim} dimensions...")
    encoder = Qwen3Encoder(output_dim=args.output_dim)
    
    # Initialize vector store
    config = {
        "provider": "chromadb",
        "collection_name": "repair_guides_qwen3",
        "database": "mist"
    }
    vector_store = ChromaVectorStore(config)
    
    # Re-index errored procedures
    reindex_errored(
        encoder=encoder,
        vector_store=vector_store,
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )


if __name__ == "__main__":
    main()

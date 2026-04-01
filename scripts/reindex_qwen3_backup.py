#!/usr/bin/env python3
"""
Re-index repair guides using Qwen3 embeddings.

Usage:
    python reindex_qwen3.py --batch-size 100 --max-docs 10000
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Iterator, Dict, Any
import logging
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.qwen3_encoder import Qwen3Encoder
from src.retrieval.chroma_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_repair_guides_from_ista(batch_size: int = 100) -> Iterator[Dict[str, Any]]:
    """
    Fetch repair guides from BMW ISTA database.
    
    Yields dicts with: id, title, text, procedure_id, etc.
    """
    from src.database.ista_db import IstaDatabase
    from src.database.xml_content import XmlContentFetcher
    import sqlite3
    
    # Use correct database path with proper case
    db_path = Path(__file__).parent.parent / "data" / "databases" / "DiagDocDb_DECRYPTED.sqlite"
    ista_db = IstaDatabase(db_path=db_path)
    xml_fetcher = XmlContentFetcher()
    
    # Get all procedure IDs directly from ISTA DB
    logger.info("Fetching procedure IDs from ISTA database...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT ID FROM XEP_INFOOBJECTS')
    procedure_ids = [str(r[0]) for r in cursor.fetchall()]
    conn.close()
    logger.info(f"Found {len(procedure_ids)} procedures")
    
    success_count = 0
    error_count = 0
    for i, proc_id in enumerate(procedure_ids):
        try:
            # Get procedure info
            info = ista_db.get_info_object(proc_id)
            if not info:
                error_count += 1
                if i < 5:
                    logger.info(f"No info for {proc_id}")
                continue
            
            title = info.get('TITLE_ENGB', '') or info.get('title', '')
            
            # Get XML content (pass title as required by the method)
            xml_content = xml_fetcher.get_content(proc_id, title)
            if not xml_content:
                error_count += 1
                if i < 5:
                    logger.debug(f"No XML content for {proc_id}")
                continue
            
            success_count += 1
            if success_count % 1000 == 0:
                logger.info(f"Processed {success_count} documents successfully ({error_count} errors)")
            
            # Get fault codes
            fault_codes = ista_db.get_fault_codes_for_procedure(proc_id)
            
            if success_count <= 3:
                logger.info(f"First document: {title[:50]}... (ID: {proc_id})")
            
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
            continue


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

def reindex_collection(
    encoder: Qwen3Encoder,
    vector_store: ChromaVectorStore,
    batch_size: int = 100,
    max_docs: int = None
):
    """
    Re-index all repair guides with Qwen3 embeddings.
    Supports resuming if interrupted.
    """
    # Use existing collection (should be created by prepare_reindex.py)
    new_collection = "repair_guides_qwen3"
    
    collection = vector_store.client.get_collection(new_collection)
    logger.info(f"Using collection: {new_collection}")
    
    # Load checkpoint to resume if needed
    already_indexed = load_checkpoint()
    if already_indexed:
        logger.info(f"Resuming from checkpoint: {len(already_indexed)} already indexed")
    
    # Fetch and index documents
    total_indexed = len(already_indexed)
    batch_ids = []
    batch_texts = []
    batch_metadatas = []
    
    for doc in tqdm(get_repair_guides_from_ista(batch_size), desc="Indexing", total=377405, initial=total_indexed):
        if max_docs and total_indexed >= max_docs:
            break
        
        # Skip if already indexed (for resume)
        if doc['id'] in already_indexed:
            continue
        
        # Prepare text for embedding
        text = f"{doc['title']}\n{doc['text']}" if doc['title'] else doc['text']
        
        # Truncate if too long (Qwen3 has token limit)
        text = text[:8000]  # Approximate token limit
        
        batch_ids.append(doc['id'])
        batch_texts.append(text)
        
        # Build metadata
        meta = {
            "title": doc['title'] or "",
            "procedure_id": doc['procedure_id'] or "",
            "procedure_name": doc['procedure_name'] or "",
            "ecu_category": doc['ecu_category'] or "",
            "text": text[:2000]  # Store first 2000 chars for display
        }
        
        # Add fault codes if present
        if doc.get('fault_codes'):
            meta['fault_codes'] = doc['fault_codes']
        else:
            meta['fault_codes'] = ['P0000']  # Placeholder for ChromaDB
        
        batch_metadatas.append(meta)
        
        # Process batch
        if len(batch_ids) >= batch_size:
            try:
                # Encode batch
                embeddings = encoder.encode(batch_texts, is_query=False)
                
                # Add to collection
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    metadatas=batch_metadatas
                )
                
                total_indexed += len(batch_ids)
                already_indexed.update(batch_ids)
                
                if total_indexed % 500 == 0:
                    logger.info(f"Indexed {total_indexed} documents...")
                    save_checkpoint(already_indexed)
                
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
            
            # Clear batch
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
            save_checkpoint(already_indexed)
        except Exception as e:
            logger.error(f"Error indexing final batch: {e}")
    
    logger.info(f"Finished indexing {total_indexed} documents to {new_collection}")
    
    # Verify
    count = collection.count()
    logger.info(f"Collection now has {count} documents")
    
    # Final checkpoint save
    save_checkpoint(already_indexed)
    logger.info(f"Checkpoint saved: {len(already_indexed)} total indexed")


def main():
    parser = argparse.ArgumentParser(description="Re-index repair guides with Qwen3")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for indexing")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to index (for testing)")
    parser.add_argument("--output-dim", type=int, default=4096, help="Output dimensions (4096, 1024, 768)")
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
        "collection_name": "repair_guides_qwen3",  # New collection
        "database": "mist"
    }
    vector_store = ChromaVectorStore(config)
    
    # Re-index
    reindex_collection(
        encoder=encoder,
        vector_store=vector_store,
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )


if __name__ == "__main__":
    main()

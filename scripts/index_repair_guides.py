#!/usr/bin/env python3
"""
Index repair guides from BMW ISTA database into vector store.

This script loads repair procedures from the ISTA database, encodes their content,
and stores embeddings in the ChromaDB vector store for semantic search.
"""
import sys
import argparse
import json
import time
import signal
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.ista_db import IstaDatabase
from database.xml_content import XmlContentFetcher
from embeddings.fault_code_encoder import FaultCodeEncoder
from retrieval.vector_store import VectorStore
from paths import get_paths
import yaml
import logging
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepairGuideIndexer:
    """Indexes repair guides from ISTA database into vector store."""
    
    def __init__(
        self,
        embedding_config: Dict[str, Any],
        retrieval_config: Dict[str, Any],
        checkpoint_file: Optional[Path] = None,
        batch_size: int = 100
    ):
        """
        Initialize repair guide indexer.
        
        Args:
            embedding_config: Embedding configuration dict
            retrieval_config: Retrieval configuration dict
            checkpoint_file: Path to checkpoint file for resume functionality
            batch_size: Batch size for encoding and storage
        """
        self.embedding_config = embedding_config
        self.retrieval_config = retrieval_config
        self.checkpoint_file = checkpoint_file or Path(__file__).parent.parent / "data" / "indexing_checkpoint.json"
        self.batch_size = batch_size
        
        # Initialize components
        logger.info("Initializing components...")
        self.ista_db = IstaDatabase()
        self.xml_fetcher = XmlContentFetcher()
        self.encoder = self._init_encoder()
        self.vector_store = VectorStore(retrieval_config["vector_store"])
        
        # Track progress
        self.indexed_ids: Set[str] = set()
        self.processed_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False

        # Chunking: ~1200 chars ≈ 300 tokens (encoder max 512)
        self.chunk_chars = 1200
        self.chunk_overlap = 200
    
    def _init_encoder(self) -> FaultCodeEncoder:
        """Initialize fault code encoder from config."""
        fault_config = self.embedding_config.get("models", {}).get("fault_code", {})
        model_name = fault_config.get("model_name", "intfloat/e5-mistral-7b-instruct")
        projection_dim = fault_config.get("projection_dim", 768)
        device = fault_config.get("device", "auto")
        
        # Resolve "auto" to actual device for logging
        if device == "auto":
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing encoder: {model_name} (dim={projection_dim}, device=auto -> {resolved_device})")
        else:
            logger.info(f"Initializing encoder: {model_name} (dim={projection_dim}, device={device})")
        
        return FaultCodeEncoder(
            model_name=model_name,
            device=device,
            projection_dim=projection_dim
        )
    
    def _load_checkpoint(self) -> None:
        """Load checkpoint file with indexed procedure IDs."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.indexed_ids = set(checkpoint.get("indexed_ids", []))
                    self.processed_count = checkpoint.get("processed_count", 0)
                    logger.info(f"Loaded checkpoint: {len(self.indexed_ids)} procedures already indexed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                self.indexed_ids = set()
        else:
            logger.info("No checkpoint found. Starting fresh.")
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint file with indexed procedure IDs."""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "indexed_ids": list(self.indexed_ids),
                "processed_count": self.processed_count,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Saved checkpoint: {len(self.indexed_ids)} procedures indexed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received. Saving progress...")
        self.shutdown_requested = True
    
    def _get_all_procedures(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all repair procedures from database.
        
        Args:
            limit: Optional limit on number of procedures to return
        
        Returns:
            List of procedure dictionaries
        """
        try:
            from sqlalchemy import text
            
            with self.ista_db.connection.session() as session:
                
                query = """
                    SELECT DISTINCT io.ID, io.TITLE_ENGB, io.NAME
                    FROM XEP_INFOOBJECTS io
                    WHERE io.ID IS NOT NULL
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = session.execute(text(query))
                rows = result.fetchall()
                
                procedures = []
                for row in rows:
                    proc_id = str(row.ID) if row.ID else None
                    if not proc_id:
                        continue
                    
                    procedures.append({
                        "id": proc_id,
                        "title_engb": str(row.TITLE_ENGB) if row.TITLE_ENGB else "",
                        "name": str(row.NAME) if row.NAME else ""
                    })
                
                return procedures
        except Exception as e:
            logger.error(f"Error querying procedures: {e}")
            raise
    
    def _get_procedure_text(self, procedure: Dict[str, Any]) -> str:
        """
        Get full text content for a procedure (title + xml content).
        
        Fetches from xmlvalueprimitive via FTS when available; otherwise title only.
        
        Args:
            procedure: Procedure dictionary with id, title_engb, name
        
        Returns:
            Combined text string (title + full content)
        """
        procedure_id = str(procedure["id"])
        title = procedure.get("title_engb") or procedure.get("name", "") or ""
        
        text_parts = [title] if title else []
        
        # Fetch full content from xmlvalueprimitive (FTS search by title)
        xml_content = self.xml_fetcher.get_content(procedure_id, title)
        if xml_content:
            text_parts.append(xml_content)
        
        return "\n\n".join(text_parts)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks for embedding."""
        if len(text) <= self.chunk_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _process_procedure(self, procedure: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single procedure: get content, chunk if needed, prepare documents.
        
        Args:
            procedure: Procedure dictionary
        
        Returns:
            List of document dicts (one per chunk) or None if error
        """
        procedure_id = str(procedure["id"])
        
        try:
            text_content = self._get_procedure_text(procedure)
            if not text_content.strip():
                logger.warning(f"Procedure {procedure_id} has no content. Skipping.")
                return None
            
            fault_codes = self.ista_db.get_fault_codes_for_procedure(procedure_id)
            title = procedure.get("title_engb") or procedure.get("name", "")
            name = procedure.get("name", "")
            
            chunks = self._chunk_text(text_content)
            documents = []
            for i, chunk_text in enumerate(chunks):
                doc_id = f"{procedure_id}_chunk_{i}" if len(chunks) > 1 else procedure_id
                documents.append({
                    "id": doc_id,
                    "text": chunk_text,
                    "title": title,
                    "procedure_id": procedure_id,
                    "procedure_name": name,
                    "fault_codes": fault_codes,
                    "ecu_category": "",
                    "metadata": {
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "indexed_at": datetime.now().isoformat(),
                    }
                })
            return documents
            
        except Exception as e:
            logger.error(f"Error processing procedure {procedure_id}: {e}", exc_info=True)
            return None
    
    def _encode_batch(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode a batch of documents.
        
        Args:
            documents: List of document dicts
        
        Returns:
            numpy array of embeddings (n_docs, dim)
        """
        texts = [doc["text"] for doc in documents]
        
        # Encode using FaultCodeEncoder (is_query=False for documents/passages)
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, normalize=True, is_query=False)
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _store_batch(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Store a batch of documents in vector store.
        
        Args:
            documents: List of document dicts
            embeddings: numpy array of embeddings
        """
        try:
            self.vector_store.add(embeddings, documents, batch_size=self.batch_size)
            
            # Track indexed procedure IDs (for resume; chunks share procedure_id)
            for doc in documents:
                self.indexed_ids.add(doc.get("procedure_id", doc["id"]))
            
            logger.debug(f"Stored batch of {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error storing batch: {e}", exc_info=True)
            raise
    
    def index(
        self,
        limit: Optional[int] = None,
        resume: bool = True,
        progress_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Index repair guides from database.
        
        Args:
            limit: Optional limit on number of procedures to index
            resume: Whether to skip already-indexed procedures
            progress_interval: Log progress every N procedures
        
        Returns:
            Dictionary with indexing statistics
        """
        logger.info("Starting repair guide indexing...")
        
        # Get all procedures
        procedures = self._get_all_procedures(limit=limit)
        total_procedures = len(procedures)
        
        logger.info(f"Found {total_procedures} procedures to process")
        if resume and self.indexed_ids:
            logger.info(f"Skipping {len(self.indexed_ids)} already-indexed procedures")
        
        # Filter out already-indexed procedures if resuming
        if resume:
            procedures = [p for p in procedures if p["id"] not in self.indexed_ids]
        
        remaining = len(procedures)
        logger.info(f"Processing {remaining} procedures")
        
        # Process in batches
        batch_documents = []
        
        for idx, procedure in enumerate(procedures):
            if self.shutdown_requested:
                logger.info("Shutdown requested. Saving progress...")
                break
            
            try:
                documents = self._process_procedure(procedure)
                
                if documents is None:
                    self.error_count += 1
                    self.processed_count += 1
                    continue
                
                for doc in documents:
                    batch_documents.append(doc)
                self.processed_count += 1
                
                # Process batch when full
                if len(batch_documents) >= self.batch_size:
                    try:
                        embeddings = self._encode_batch(batch_documents)
                        self._store_batch(batch_documents, embeddings)
                        batch_documents = []
                    except Exception as e:
                        logger.error(f"Error storing batch: {e}", exc_info=True)
                        # Don't count batch storage failures as document errors
                        # The documents will be retried on next run (not in indexed_ids)
                        # Clear batch to avoid double-processing
                        batch_documents = []
                
                # Log progress
                if (idx + 1) % progress_interval == 0:
                    elapsed = time.time() - self.start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    remaining_procs = remaining - (idx + 1)
                    eta = remaining_procs / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {idx + 1}/{remaining} procedures "
                        f"({(idx + 1) / remaining * 100:.1f}%) | "
                        f"Rate: {rate:.1f} proc/s | "
                        f"ETA: {eta / 60:.1f} min | "
                        f"Errors: {self.error_count}"
                    )
                    
                    # Save checkpoint periodically
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error processing procedure {procedure.get('id', 'unknown')}: {e}", exc_info=True)
                self.error_count += 1
                continue
        
        # Process remaining batch
        if batch_documents and not self.shutdown_requested:
            try:
                embeddings = self._encode_batch(batch_documents)
                self._store_batch(batch_documents, embeddings)
            except Exception as e:
                logger.error(f"Error storing final batch: {e}", exc_info=True)
                # Don't count batch storage failures as document errors
                # The documents will be retried on next run (not in indexed_ids)
        
        # Final checkpoint save
        self._save_checkpoint()
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        
        stats = {
            "total_procedures": total_procedures,
            "processed": self.processed_count,
            "indexed": len(self.indexed_ids),
            "errors": self.error_count,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_minutes": elapsed_time / 60,
            "rate_per_second": self.processed_count / elapsed_time if elapsed_time > 0 else 0
        }
        
        logger.info("Indexing complete!")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        return stats
    
    def close(self):
        """Close database connections."""
        if hasattr(self, "xml_fetcher"):
            self.xml_fetcher.close()


def main():
    """Main entry point for indexing script."""
    parser = argparse.ArgumentParser(
        description="Index repair guides from BMW ISTA database into vector store"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, ignoring checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for encoding and storage (default: 100)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of procedures to index (for testing)"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to checkpoint file (default: data/indexing_checkpoint.json)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Log progress every N procedures (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Get paths
    paths = get_paths()
    
    # Load configs
    logger.info("Loading configurations...")
    with open(paths.embedding_config, 'r') as f:
        embedding_config = yaml.safe_load(f)
    
    with open(paths.retrieval_config, 'r') as f:
        retrieval_config = yaml.safe_load(f)
    
    # Initialize indexer
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else None
    indexer = RepairGuideIndexer(
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
        checkpoint_file=checkpoint_file,
        batch_size=args.batch_size
    )
    
    try:
        # Run indexing
        stats = indexer.index(
            limit=args.limit,
            resume=args.resume,
            progress_interval=args.progress_interval
        )
        
        # Print summary
        print("\n" + "="*60)
        print("INDEXING SUMMARY")
        print("="*60)
        print(f"Total procedures found: {stats['total_procedures']}")
        print(f"Procedures processed: {stats['processed']}")
        print(f"Procedures indexed: {stats['indexed']}")
        print(f"Errors: {stats['errors']}")
        print(f"Elapsed time: {stats['elapsed_time_minutes']:.1f} minutes")
        print(f"Rate: {stats['rate_per_second']:.2f} procedures/second")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving progress...")
        indexer._save_checkpoint()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        indexer._save_checkpoint()
        sys.exit(1)
    finally:
        indexer.close()


if __name__ == "__main__":
    main()

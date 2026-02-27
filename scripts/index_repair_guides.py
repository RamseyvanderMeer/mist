#!/usr/bin/env python3
"""
Index repair guides from BMW ISTA database into vector store.

This script loads repair procedures from the ISTA database, encodes their content,
and stores embeddings in the ChromaDB vector store for semantic search.

Multi-machine mode (when DATABASE_URL is set):
  1. Run migration: python scripts/run_indexing_work_migration.py
  2. Start workers: python scripts/index_repair_guides.py --worker-id machine-1 --batch-size 512
     (Workers auto-seed from ISTA when queue is empty; no separate --seed step needed.)
  3. --no-resume: Truncate and re-seed the queue for a fresh start.
"""
import os
import sys
import argparse
import json
import time
import signal
import socket
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.exc import OperationalError

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _ensure_indexing_work_table(engine) -> None:
    """Create indexing_work table if it does not exist."""
    migration_file = Path(__file__).parent.parent / "scripts" / "migrations" / "create_indexing_work_postgres.sql"
    if not migration_file.exists():
        raise FileNotFoundError(f"Migration file not found: {migration_file}")
    from sqlalchemy import text
    with open(migration_file, "r") as f:
        sql = f.read()
    with engine.connect() as conn:
        for stmt in (s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")):
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise


def _seed_indexing_work(engine, procedure_ids: List[str]) -> int:
    """Insert procedure IDs into indexing_work as pending. Returns count of new rows added."""
    from sqlalchemy import text
    stmt = text("""
        INSERT INTO indexing_work (procedure_id, status)
        VALUES (:pid, 'pending')
        ON CONFLICT (procedure_id) DO NOTHING
    """)
    added = 0
    with engine.connect() as conn:
        for pid in procedure_ids:
            try:
                result = conn.execute(stmt, {"pid": pid})
                conn.commit()
                if result.rowcount and result.rowcount > 0:
                    added += 1
            except Exception:
                conn.rollback()
    return added


def _truncate_and_seed(engine, procedure_ids: List[str]) -> int:
    """Truncate indexing_work and insert all procedure IDs. Returns total rows inserted."""
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE indexing_work"))
        conn.commit()
    # Insert in batches (batch commits for speed)
    stmt = text("INSERT INTO indexing_work (procedure_id, status) VALUES (:pid, 'pending')")
    chunk_size = 5000
    with engine.connect() as conn:
        for i in range(0, len(procedure_ids), chunk_size):
            chunk = procedure_ids[i : i + chunk_size]
            for pid in chunk:
                conn.execute(stmt, {"pid": pid})
            conn.commit()
    return len(procedure_ids)


def _claim_batch(engine, worker_id: str, batch_size: int) -> List[str]:
    """Claim a batch of pending procedures. Returns list of procedure_ids."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Use FOR UPDATE SKIP LOCKED so multiple workers get different rows
        result = conn.execute(
            text("""
                WITH claimed AS (
                    SELECT procedure_id FROM indexing_work
                    WHERE status = 'pending'
                    LIMIT :batch_size
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE indexing_work iw
                SET status = 'in_progress', worker_id = :worker_id, started_at = NOW()
                FROM claimed c
                WHERE iw.procedure_id = c.procedure_id
                RETURNING iw.procedure_id
            """),
            {"worker_id": worker_id, "batch_size": batch_size}
        )
        ids = [row[0] for row in result]
        conn.commit()
        return ids


def _mark_completed(engine, procedure_id: str) -> None:
    """Mark a procedure as completed."""
    _mark_completed_batch(engine, [procedure_id])


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)
def _mark_completed_batch(engine, procedure_ids: List[str]) -> None:
    """Mark multiple procedures as completed in one DB round-trip. Retries on connection errors."""
    if not procedure_ids:
        return
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(
            text("""
                UPDATE indexing_work
                SET status = 'completed', completed_at = NOW()
                WHERE procedure_id = ANY(:pids)
            """),
            {"pids": procedure_ids}
        )
        conn.commit()


def _mark_failed(engine, procedure_id: str, error_message: str) -> None:
    """Mark a procedure as failed."""
    _mark_failed_batch(engine, [procedure_id], error_message)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)
def _mark_failed_batch(engine, procedure_ids: List[str], error_message: str) -> None:
    """Mark multiple procedures as failed in one DB round-trip. Retries on connection errors."""
    if not procedure_ids:
        return
    from sqlalchemy import text
    err = (error_message or "")[:1000]
    with engine.connect() as conn:
        conn.execute(
            text("""
                UPDATE indexing_work
                SET status = 'failed', completed_at = NOW(), error_message = :err
                WHERE procedure_id = ANY(:pids)
            """),
            {"pids": procedure_ids, "err": err}
        )
        conn.commit()


def _get_pending_count(engine) -> int:
    """Return count of pending procedures."""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM indexing_work WHERE status = 'pending'"))
        return result.scalar() or 0


def _reset_stuck(engine, older_than_minutes: int = 60) -> int:
    """Reset in_progress to pending for rows older than threshold (e.g. crashed workers)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                UPDATE indexing_work
                SET status = 'pending', worker_id = NULL, started_at = NULL
                WHERE status = 'in_progress'
                AND started_at < NOW() - INTERVAL '1 minute' * :mins
            """),
            {"mins": older_than_minutes}
        )
        conn.commit()
        return result.rowcount or 0


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
    
    def _get_procedure_by_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get a single procedure by ID. Returns None if not found."""
        obj = self.ista_db.get_info_object(procedure_id)
        if not obj:
            return None
        return {
            "id": str(obj.get("ID", procedure_id)),
            "title_engb": str(obj.get("TITLE_ENGB", "") or ""),
            "name": str(obj.get("NAME", "") or ""),
        }

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
            fault_labels = self.ista_db.get_fault_labels_for_procedure(procedure_id)
            title = procedure.get("title_engb") or procedure.get("name", "")
            name = procedure.get("name", "")

            # Include DB fault codes and labels in text for semantic search
            # - Fault codes: align with query format
            # - Fault labels: problem/symptom descriptions (e.g. "Engine Coolant Temperature Sensor
            #   Circuit Malfunction") help match user symptoms like "engine too hot" to procedures
            prefix_parts = []
            if fault_codes:
                prefix_parts.append(f"Fault codes: {', '.join(fault_codes)}.")
            if fault_labels:
                prefix_parts.append(f"Problem descriptions: {'; '.join(fault_labels[:5])}.")
            fault_prefix = "\n\n".join(prefix_parts) + "\n\n" if prefix_parts else ""
            
            chunks = self._chunk_text(text_content)
            documents = []
            for i, chunk_text in enumerate(chunks):
                doc_id = f"{procedure_id}_chunk_{i}" if len(chunks) > 1 else procedure_id
                # Prepend fault codes to text for encoding (aligns with query format)
                text_for_embedding = fault_prefix + chunk_text
                documents.append({
                    "id": doc_id,
                    "text": text_for_embedding,
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
        
        # Clear progress when --no-resume (fresh start)
        if not resume:
            self.indexed_ids = set()
            self.processed_count = 0
            if self.checkpoint_file.exists():
                try:
                    self.checkpoint_file.unlink()
                    logger.info("Cleared checkpoint for fresh start (--no-resume)")
                except Exception as e:
                    logger.warning(f"Could not delete checkpoint file: {e}")
        
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

    def index_with_db(
        self,
        db_url: str,
        worker_id: str,
        seed: bool = False,
        seed_only: bool = False,
        reset_stuck_minutes: Optional[int] = None,
        progress_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Index using shared PostgreSQL work queue (multi-machine safe).

        Args:
            db_url: PostgreSQL connection URL
            worker_id: Unique identifier for this worker
            seed: If True, populate work queue from ISTA before indexing
            seed_only: If True, only seed and exit
            reset_stuck_minutes: If set, reset in_progress older than N min to pending
            progress_interval: Log progress every N procedures

        Returns:
            Dictionary with indexing statistics
        """
        from sqlalchemy import create_engine
        engine = create_engine(
            db_url,
            pool_pre_ping=True,  # Test connection before use (avoids stale SSL)
            pool_recycle=300,    # Recycle connections every 5 min (NeonDB idle timeout)
        )
        _ensure_indexing_work_table(engine)

        if seed or seed_only:
            procedures = self._get_all_procedures()
            procedure_ids = [p["id"] for p in procedures]
            logger.info(f"Seeding indexing_work with {len(procedure_ids)} procedure IDs...")
            added = _seed_indexing_work(engine, procedure_ids)
            logger.info(f"Seeded {added} new procedures (existing skipped)")
            if seed_only:
                return {"seeded": added, "total": len(procedure_ids)}

        if reset_stuck_minutes is not None:
            reset = _reset_stuck(engine, reset_stuck_minutes)
            if reset:
                logger.info(f"Reset {reset} stuck in_progress rows to pending")

        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0

        total_indexed = 0
        total_errors = 0

        while not self.shutdown_requested:
            claimed = _claim_batch(engine, worker_id, self.batch_size)
            if not claimed:
                pending = _get_pending_count(engine)
                if pending == 0:
                    # Auto-seed if queue is empty (no separate --seed step needed)
                    procedures = self._get_all_procedures()
                    procedure_ids = [p["id"] for p in procedures]
                    added = _seed_indexing_work(engine, procedure_ids)
                    if added > 0:
                        logger.info("Auto-seeded %d procedures (queue was empty)", added)
                        continue
                    logger.info("No more work. All procedures indexed.")
                    break
                logger.debug(f"No work claimed (others may be processing). Pending: {pending}. Retrying...")
                time.sleep(2)
                continue

            batch_documents = []
            for procedure_id in claimed:
                try:
                    procedure = self._get_procedure_by_id(procedure_id)
                    if not procedure:
                        _mark_failed(engine, procedure_id, "Procedure not found in ISTA")
                        total_errors += 1
                        continue

                    documents = self._process_procedure(procedure)
                    if documents is None:
                        _mark_failed(engine, procedure_id, "Processing failed")
                        total_errors += 1
                        continue

                    for doc in documents:
                        batch_documents.append(doc)
                    self.processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing {procedure_id}: {e}", exc_info=True)
                    _mark_failed(engine, procedure_id, str(e))
                    total_errors += 1

            if batch_documents:
                try:
                    embeddings = self._encode_batch(batch_documents)
                    self._store_batch(batch_documents, embeddings)
                    completed_ids = set()
                    for doc in batch_documents:
                        pid = str(doc.get("procedure_id", doc["id"]))
                        if "_chunk_" in pid:
                            pid = pid.split("_chunk_")[0]
                        completed_ids.add(pid)
                    _mark_completed_batch(engine, list(completed_ids))
                    total_indexed += len(completed_ids)
                except Exception as e:
                    logger.error(f"Error storing batch: {e}", exc_info=True)
                    _mark_failed_batch(engine, claimed, str(e))

            if self.processed_count % progress_interval == 0:
                elapsed = time.time() - self.start_time
                rate = self.processed_count / elapsed if elapsed > 0 else 0
                pending = _get_pending_count(engine)
                logger.info(
                    f"Worker {worker_id}: {self.processed_count} processed | "
                    f"Indexed: {total_indexed} | Errors: {total_errors} | "
                    f"Pending: {pending} | Rate: {rate:.1f}/s"
                )

        elapsed_time = time.time() - self.start_time
        stats = {
            "processed": self.processed_count,
            "indexed": total_indexed,
            "errors": total_errors,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_minutes": elapsed_time / 60,
            "rate_per_second": self.processed_count / elapsed_time if elapsed_time > 0 else 0,
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
        help="Start fresh. Local: clear checkpoint. Multi-machine: truncate work queue and re-seed from ISTA."
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
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID for multi-machine mode (default: hostname). Requires DATABASE_URL."
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Populate work queue from ISTA before indexing (multi-machine mode)"
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only populate work queue and exit (run once before starting workers)"
    )
    parser.add_argument(
        "--reset-stuck",
        type=int,
        metavar="MINUTES",
        default=4,
        help="Reset in_progress rows older than N minutes to pending (for crashed workers)"
    )

    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    use_db = bool(db_url and db_url.startswith("postgresql"))

    worker_id = args.worker_id or socket.gethostname() or f"worker-{os.getpid()}"
    if use_db:
        logger.info(f"Multi-machine mode: worker_id={worker_id}")

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
        if use_db:
            if args.seed_only and not args.seed:
                args.seed = True
            if not args.resume:
                from sqlalchemy import create_engine
                engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                )
                _ensure_indexing_work_table(engine)
                procedures = indexer._get_all_procedures()
                procedure_ids = [p["id"] for p in procedures]
                inserted = _truncate_and_seed(engine, procedure_ids)
                logger.info("Reset and seeded %d procedures (--no-resume)", inserted)
            stats = indexer.index_with_db(
                db_url=db_url,
                worker_id=worker_id,
                seed=args.seed,
                seed_only=args.seed_only,
                reset_stuck_minutes=args.reset_stuck,
                progress_interval=args.progress_interval,
            )
            print("\n" + "=" * 60)
            print("INDEXING SUMMARY (multi-machine)")
            print("=" * 60)
            if "seeded" in stats:
                print(f"Seeded: {stats['seeded']} new procedures")
                print(f"Total in queue: {stats.get('total', 'N/A')}")
            else:
                print(f"Procedures processed: {stats['processed']}")
                print(f"Procedures indexed: {stats['indexed']}")
                print(f"Errors: {stats['errors']}")
                print(f"Elapsed time: {stats['elapsed_time_minutes']:.1f} minutes")
                print(f"Rate: {stats['rate_per_second']:.2f} procedures/second")
            print("=" * 60)
        else:
            if args.seed or args.seed_only:
                logger.warning("DATABASE_URL not set. --seed/--seed-only require PostgreSQL. Using local mode.")
            stats = indexer.index(
                limit=args.limit,
                resume=args.resume,
                progress_interval=args.progress_interval
            )
            print("\n" + "=" * 60)
            print("INDEXING SUMMARY")
            print("=" * 60)
            print(f"Total procedures found: {stats['total_procedures']}")
            print(f"Procedures processed: {stats['processed']}")
            print(f"Procedures indexed: {stats['indexed']}")
            print(f"Errors: {stats['errors']}")
            print(f"Elapsed time: {stats['elapsed_time_minutes']:.1f} minutes")
            print(f"Rate: {stats['rate_per_second']:.2f} procedures/second")
            print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving progress...")
        if not use_db:
            indexer._save_checkpoint()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if not use_db:
            indexer._save_checkpoint()
        sys.exit(1)
    finally:
        indexer.close()


if __name__ == "__main__":
    main()

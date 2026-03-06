"""
ChromaDB vector store for repair guide embeddings.

Uses ChromaDB Cloud (CloudClient) with CHROMA_DB_API_KEY and CHROMA_DB_TENANT from .env.
"""
from typing import List, Dict, Optional, Union, Any
import numpy as np
import logging
import os
import uuid

from .vector_store import (
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreConfigurationError,
    VectorStoreOperationError,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store for repair guide embeddings.

    Implements add, search, update, delete, scroll, and get_collection_info.
    Uses ChromaDB Cloud when CHROMA_DB_API_KEY and CHROMA_DB_TENANT are set.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize ChromaDB vector store.

        Args:
            config: Configuration dict with:
                - provider: "chromadb"
                - collection_name: Collection name
                - distance_metric: "cosine", "euclidean", or "l2"
                - vector_size: Vector dimension (default: 1024)
                - database: ChromaDB database name (default: "mist")

        Environment Variables:
            CHROMA_DB_API_KEY: ChromaDB Cloud API key
            CHROMA_DB_TENANT: ChromaDB Cloud tenant ID
        """
        self.config = config
        self.collection_name = config.get("collection_name", "repair_guides")
        self.vector_size = config.get("vector_size", 1024)
        self.database = config.get("database", "mist")

        api_key = os.getenv("CHROMA_DB_API_KEY") or config.get("api_key")
        tenant = os.getenv("CHROMA_DB_TENANT") or config.get("tenant")

        if not api_key or not tenant:
            raise VectorStoreConfigurationError(
                "ChromaDB Cloud requires CHROMA_DB_API_KEY and CHROMA_DB_TENANT "
                "(or api_key/tenant in config)"
            )

        try:
            import chromadb

            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=self.database,
            )
            logger.info(
                f"Connected to ChromaDB Cloud (database={self.database}, "
                f"collection={self.collection_name})"
            )
        except ImportError as e:
            raise VectorStoreConfigurationError(
                "chromadb package required for ChromaDB provider. Install with: pip install chromadb"
            ) from e
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise VectorStoreConnectionError(f"Failed to connect to ChromaDB: {e}") from e

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            metric = self.config.get("distance_metric", "cosine").lower()
            # ChromaDB: "cosine", "l2", "ip"
            if metric == "euclidean" or metric == "euclid":
                metric = "l2"
            elif metric == "dot":
                metric = "ip"

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric},
            )
        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to create/get collection: {e}"
            ) from e

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Add documents to ChromaDB."""
        if len(embeddings) != len(documents):
            raise VectorStoreOperationError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of documents ({len(documents)})"
            )

        try:
            ids = []
            metadatas = []
            emb_list = []

            for i, (emb, doc) in enumerate(zip(embeddings, documents)):
                doc_id = doc.get("id", i)
                if isinstance(doc_id, (int, str)) and not isinstance(doc_id, uuid.UUID):
                    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
                    doc_id = str(uuid.uuid5(namespace, str(doc_id)))

                ids.append(str(doc_id))
                emb_list.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))

                # ChromaDB metadata: strings, numbers, bools, or lists of those
                fault_codes = doc.get("fault_codes", [])
                metadata = {
                    "text": (doc.get("text", "") or "")[:50000],  # ChromaDB limit
                    "title": str(doc.get("title", "") or ""),
                    "procedure_id": str(doc.get("procedure_id", "") or ""),
                    "procedure_name": str(doc.get("procedure_name", "") or ""),
                    "ecu_category": str(doc.get("ecu_category", "") or ""),
                }
                if fault_codes:
                    metadata["fault_codes"] = fault_codes
                meta_extra = doc.get("metadata", {})
                if meta_extra:
                    for k, v in meta_extra.items():
                        if isinstance(v, (str, int, float, bool, list)):
                            metadata[f"meta_{k}"] = v
                metadatas.append(metadata)

            # ChromaDB Cloud limits upsert to 300 per request; split if needed
            upload_chunk = min(batch_size, 300)
            for i in range(0, len(ids), upload_chunk):
                batch_ids = ids[i : i + upload_chunk]
                batch_emb = emb_list[i : i + upload_chunk]
                batch_meta = metadatas[i : i + upload_chunk]
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_emb,
                    metadatas=batch_meta,
                )

            logger.info(f"Added {len(ids)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise VectorStoreOperationError(f"Failed to add documents: {e}") from e

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            query_vec = query_embedding.tolist()

            return self._search_with_filters(
                query_vec=query_vec,
                top_k=top_k,
                filter_dict=filter_dict,
            )
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise VectorStoreOperationError(f"Search failed: {e}") from e

    def _search_with_filters(
        self,
        query_vec: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute search with fault-code filters while staying within Chroma predicate limits.

        Some Chroma Cloud plans cap where-clause predicates (default 8). If a
        fault_codes list exceeds that, we run batched queries and merge results.
        """
        where_filters = self._build_where(filter_dict)
        normalized_filter: Optional[Dict[str, Any]] = dict(filter_dict) if filter_dict else None

        fault_codes = None
        if filter_dict and "fault_codes" in filter_dict:
            raw_codes = filter_dict.get("fault_codes")
            if isinstance(raw_codes, list):
                seen = set()
                fault_codes = []
                for code in raw_codes:
                    if not code:
                        continue
                    code_str = str(code).strip()
                    if not code_str:
                        continue
                    if code_str in seen:
                        continue
                    seen.add(code_str)
                    fault_codes.append(code_str)

            if normalized_filter is not None:
                normalized_filter["fault_codes"] = fault_codes
            where_filters = self._build_where(normalized_filter)

        if not fault_codes:
            return self._execute_query(query_vec=query_vec, top_k=top_k, where=where_filters)

        try:
            max_predicates = int(os.getenv("CHROMA_MAX_WHERE_PREDICATES", "8"))
        except Exception:
            max_predicates = 8

        # Chroma Cloud enforces a hard quota on where predicates (often 8).
        # Treat env var as a hint but cap to 8 by default so we don't trip quotas.
        hard_cap = int(os.getenv("CHROMA_WHERE_PREDICATE_HARD_CAP", "8") or "8")
        if hard_cap > 0:
            if max_predicates > hard_cap:
                logger.warning(
                    "CHROMA_MAX_WHERE_PREDICATES=%s exceeds hard cap=%s; capping to %s to avoid quota errors.",
                    max_predicates,
                    hard_cap,
                    hard_cap,
                )
                max_predicates = hard_cap

        if max_predicates <= 0:
            max_predicates = len(fault_codes)

        if len(fault_codes) <= max_predicates:
            return self._execute_query(query_vec=query_vec, top_k=top_k, where=where_filters)

        # Build non-fault filters once and split only fault-code predicates in chunks.
        logger.warning(
            "Fault code predicate count (%s) exceeds configured Chroma limit (%s); "
            "querying in chunks and merging results.",
            len(fault_codes),
            max_predicates,
        )
        base_filters = {
            key: value for key, value in (normalized_filter or {}).items() if key != "fault_codes"
        }
        base_where = self._build_where(base_filters) if base_filters else None

        merged: Dict[str, Dict[str, Any]] = {}
        for idx in range(0, len(fault_codes), max_predicates):
            chunk_codes = fault_codes[idx : idx + max_predicates]
            if not chunk_codes:
                continue

            chunk_filter = {
                "$or": [{"fault_codes": {"$contains": c}} for c in chunk_codes]
            }
            if base_where is not None:
                chunk_where = {"$and": [base_where, chunk_filter]}
            else:
                chunk_where = chunk_filter

            results = self._execute_query(query_vec=query_vec, top_k=top_k, where=chunk_where)
            for item in results:
                existing = merged.get(item["id"])
                if existing is None or item["score"] > existing["score"]:
                    merged[item["id"]] = item

        if merged:
            ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
            return ranked[:top_k]

        # If chunked queries return nothing, keep behavior same as existing
        return []

    def _execute_query(
        self,
        query_vec: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        formatted: List[Dict[str, Any]] = []
        if result and result["ids"] and result["ids"][0]:
            ids = result["ids"][0]
            metadatas = result.get("metadatas", [[]])[0] or []
            distances = result.get("distances", [[]])[0] or []

            for j, doc_id in enumerate(ids):
                meta = metadatas[j] if j < len(metadatas) else {}
                dist = distances[j] if j < len(distances) else 0.0
                # ChromaDB cosine returns distance (0=identical); convert to similarity
                score = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)
                formatted.append({
                    "id": str(doc_id),
                    "score": float(score),
                    "text": meta.get("text", ""),
                    "title": meta.get("title", ""),
                    "procedure_id": meta.get("procedure_id", ""),
                    "procedure_name": meta.get("procedure_name", ""),
                    "fault_codes": meta.get("fault_codes", []),
                    "ecu_category": meta.get("ecu_category", ""),
                })
        return formatted

    def update(
        self,
        doc_id: Union[str, int],
        embedding: Optional[np.ndarray] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update document (ChromaDB upsert with same ID)."""
        if embedding is None and payload is None:
            logger.warning(f"No updates provided for document {doc_id}")
            return
        try:
            if embedding is not None:
                meta = payload or {}
                self.collection.upsert(
                    ids=[str(doc_id)],
                    embeddings=[embedding.flatten().tolist()],
                    metadatas=[meta],
                )
            elif payload is not None:
                try:
                    self.collection.update(ids=[str(doc_id)], metadatas=[payload])
                except Exception:
                    # ChromaDB may require embedding for update; get existing and upsert
                    existing = self.collection.get(ids=[str(doc_id)], include=["embeddings"])
                    if existing and existing["embeddings"]:
                        merged = {**existing.get("metadatas", [{}])[0], **payload}
                        self.collection.upsert(
                            ids=[str(doc_id)],
                            embeddings=existing["embeddings"],
                            metadatas=[merged],
                        )
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to update {doc_id}: {e}") from e

    def delete(
        self,
        doc_id: Optional[Union[str, int]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents."""
        if doc_id is None and filter_dict is None:
            raise VectorStoreOperationError("Either doc_id or filter_dict required")
        if doc_id is not None and filter_dict is not None:
            raise VectorStoreOperationError("Cannot specify both doc_id and filter_dict")
        try:
            if doc_id is not None:
                self.collection.delete(ids=[str(doc_id)])
            else:
                where = self._build_where(filter_dict)
                if where:
                    self.collection.delete(where=where)
        except Exception as e:
            raise VectorStoreOperationError(f"Delete failed: {e}") from e

    def _build_where(self, filter_dict: Dict[str, Any]) -> Optional[Dict]:
        """Build ChromaDB where clause from filter dict."""
        if not filter_dict:
            return None
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append({"$or": [{key: {"$contains": v}} for v in value]})
            else:
                conditions.append({key: {"$eq": value}})
        return {"$and": conditions} if len(conditions) > 1 else conditions[0]

    def scroll(
        self,
        limit: int = 1000,
        offset: Optional[int] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        """
        Scroll through collection.
        Returns (points, next_offset) where points have .id and .payload.
        """
        offset = offset or 0
        include = ["metadatas"]
        if with_vectors:
            include.append("embeddings")
        result = self.collection.get(
            limit=limit,
            offset=offset,
            include=include,
        )
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [[]])
        embeddings = result.get("embeddings", [[]]) if with_vectors else None

        class Point:
            pass

        points = []
        for i, doc_id in enumerate(ids):
            p = Point()
            p.id = doc_id
            p.payload = metadatas[i] if i < len(metadatas) else {}
            if with_vectors and embeddings and i < len(embeddings):
                p.vector = embeddings[i]
            points.append(p)

        next_offset = offset + len(points) if len(points) == limit else None
        return (points, next_offset)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "vectors_count": count,
                "points_count": count,
                "config": {"vector_size": self.vector_size},
            }
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to get info: {e}") from e

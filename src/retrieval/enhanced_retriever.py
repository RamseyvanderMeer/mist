"""
Enhanced retriever orchestrator for multi-stage retrieval pipeline.

This module provides the EnhancedRetriever class that orchestrates a 4-stage
retrieval pipeline combining vector search, re-ranking, knowledge graph scoring,
and combined ranking to return top-K repair guide recommendations.
"""
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
import yaml
import numpy as np
import torch

from src.paths import Paths
from src.retrieval.vector_store import VectorStore, VectorStoreOperationError
from src.retrieval.reranker import Reranker, RerankerAPIError, RerankerModelError
from src.retrieval.ranker import Ranker, RankerError
from src.knowledge.graph_query import KnowledgeGraphQuery
from src.feedback.collector import FeedbackCollector
from src.embeddings.fault_code_encoder import FaultCodeEncoder

logger = logging.getLogger(__name__)


class EnhancedRetrieverError(Exception):
    """Base exception for EnhancedRetriever errors."""
    pass


class EnhancedRetriever:
    """
    Multi-stage retrieval orchestrator combining:
    - Vector similarity search
    - Cross-encoder re-ranking
    - Knowledge graph path scoring
    - Combined ranking with feedback
    
    Orchestrates a 4-stage pipeline:
    1. Vector search (top-K=100)
    2. Re-ranking (top-K=50)
    3. KG path scoring
    4. Combined scoring and final ranking
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize EnhancedRetriever with configuration.
        
        Args:
            config_path: Path to retrieval_config.yaml. If None, uses default from Paths.
        
        Raises:
            EnhancedRetrieverError: If initialization fails
        """
        # Load configuration
        paths = Paths()
        if config_path is None:
            config_path = paths.retrieval_config
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise EnhancedRetrieverError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise EnhancedRetrieverError(f"Failed to load config from {config_path}: {e}") from e
        
        if self.config is None:
            raise EnhancedRetrieverError(f"Config file {config_path} is empty")
        
        # Extract retrieval parameters
        retrieval_config = self.config.get("retrieval", {})
        self.initial_k = retrieval_config.get("initial_k", 100)
        self.rerank_k = retrieval_config.get("rerank_k", 50)
        self.final_k = retrieval_config.get("final_k", 10)
        
        # Initialize components
        try:
            # VectorStore
            vector_store_config = self.config.get("vector_store", {})
            self.vector_store = VectorStore(vector_store_config)
            
            # Reranker
            reranking_config = self.config.get("reranking", {})
            self.reranker = Reranker(reranking_config)
            
            # Knowledge Graph Query
            kg_config = self.config.get("knowledge_graph", {})
            kg_enabled = kg_config.get("enabled", True)
            if kg_enabled:
                kg_path = kg_config.get("graph_path", str(paths.knowledge_graph))
                self.kg_query = KnowledgeGraphQuery(kg_path)
            else:
                self.kg_query = None
                logger.info("Knowledge graph disabled in config")
            
            # Ranker
            ranking_config = self.config.get("ranking", {})
            self.ranker = Ranker(ranking_config)
            
            # FeedbackCollector (optional)
            try:
                self.feedback_collector = FeedbackCollector()
            except Exception as e:
                logger.warning(f"Failed to initialize FeedbackCollector: {e}. Continuing without feedback.")
                self.feedback_collector = None
            
            # FaultCodeEncoder - must match index (procedure text encoded with is_query=False)
            # Query uses is_query=True for E5 asymmetric retrieval
            embedding_config_path = paths.embedding_config
            embedding_config = {}
            if embedding_config_path.exists():
                try:
                    with open(embedding_config_path, 'r', encoding='utf-8') as f:
                        embedding_config = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load embedding config: {e}. Using defaults.")
            fc_config = embedding_config.get("models", {}).get("fault_code", {})
            self.encoder = FaultCodeEncoder(
                model_name=fc_config.get("model_name", "intfloat/e5-mistral-7b-instruct"),
                device=fc_config.get("device", "auto"),
                projection_dim=fc_config.get("projection_dim", 1024),
            )
            
            logger.info(
                f"Initialized EnhancedRetriever: initial_k={self.initial_k}, "
                f"rerank_k={self.rerank_k}, final_k={self.final_k}"
            )
            
        except Exception as e:
            raise EnhancedRetrieverError(f"Failed to initialize components: {e}") from e
    
    def retrieve(
        self,
        fault_codes: List[str],
        obd_data: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        description: Optional[str] = None,
        search_query_text: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-stage retrieval pipeline.
        
        Args:
            fault_codes: List of fault code strings (e.g., ["P0301", "P0302"])
            obd_data: Optional OBD sensor data dictionary
            query_text: Optional query text for re-ranking. If None, built from fault_codes + description.
            description: Optional problem/symptom description for semantic search.
            search_query_text: Override for vector search query (e.g. expanded query from clarification).
            top_k: Number of final results to return. If None, uses config final_k.
        
        Returns:
            List of ranked result dictionaries, each containing:
            - id: Document ID
            - score: Embedding similarity score
            - rerank_score: Re-ranking score (if available)
            - combined_score: Final combined score
            - procedure_id: Procedure identifier
            - text: Document text
            - title: Document title
            - Additional metadata fields
        """
        if top_k is None:
            top_k = self.final_k

        # Allow description-only retrieval (symptom-based search when no fault codes)
        has_query = (
            (fault_codes and len(fault_codes) > 0)
            or (description and description.strip())
            or (search_query_text and search_query_text.strip())
        )
        if not has_query:
            logger.warning("No fault codes or description provided, returning empty results")
            return []
        
        # Build query text for vector search and re-ranking
        if search_query_text is None:
            search_query_text = self._build_query_text(fault_codes, description)
        if query_text is None:
            query_text = search_query_text
        
        # Stage 1: Vector search (FaultCodeEncoder, same space as indexed procedure text)
        candidates = self._stage1_vector_search(search_query_text)
        if not candidates:
            logger.warning("Stage 1 returned no candidates")
            return []
        
        # Stage 2: Re-ranking
        candidates = self._stage2_reranking(candidates, fault_codes, query_text)
        if not candidates:
            logger.warning("Stage 2 returned no candidates")
            return []
        
        # Stage 3: KG path scoring
        kg_scores = self._stage3_kg_scoring(candidates, fault_codes)
        
        # Stage 4: Combined scoring
        ranked_results = self._stage4_combined_scoring(candidates, kg_scores)
        
        # Return top_k results
        return ranked_results[:top_k]
    
    def _build_query_text(
        self,
        fault_codes: List[str],
        description: Optional[str] = None
    ) -> str:
        """Build query text for vector search from fault codes and optional description."""
        if fault_codes:
            fault_text = ", ".join(fault_codes)
            if description and description.strip():
                return f"Fault codes: {fault_text}. Problem: {description.strip()}"
            return fault_text
        if description and description.strip():
            return f"Problem: {description.strip()}"
        return ""

    def _stage1_vector_search(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Stage 1: Vector search for initial candidates.
        
        Uses FaultCodeEncoder (same as index) with is_query=True for E5 asymmetric retrieval.
        Index stores procedure text with is_query=False.
        
        Args:
            query_text: Query text (fault codes + optional problem description)
        
        Returns:
            List of candidate dictionaries from vector search
        """
        try:
            logger.debug(f"Encoding query: {query_text[:100]}...")
            
            self.encoder.eval()
            with torch.no_grad():
                query_embedding = self.encoder.encode(
                    query_text, normalize=True, is_query=True
                )
            
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()
            
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            candidates = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.initial_k,
                filter_dict=None
            )
            
            logger.info(f"Stage 1: Retrieved {len(candidates)} candidates")
            return candidates
            
        except VectorStoreOperationError as e:
            logger.error(f"Stage 1 vector search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Stage 1 unexpected error: {e}", exc_info=True)
            return []
    
    def _stage2_reranking(
        self,
        candidates: List[Dict[str, Any]],
        fault_codes: List[str],
        query_text: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Re-rank candidates using cross-encoder.
        
        Args:
            candidates: List of candidates from Stage 1
            fault_codes: List of fault code strings
            query_text: Optional query text. If None, generated from fault codes.
        
        Returns:
            List of candidates with rerank_score added
        """
        if not self.reranker.enabled:
            logger.debug("Reranker disabled, skipping Stage 2")
            # Add default rerank_score of 0.0
            for candidate in candidates:
                candidate["rerank_score"] = 0.0
            return candidates[:self.rerank_k]
        
        try:
            # Generate query text if not provided
            if query_text is None:
                query_text = ", ".join(fault_codes)
            
            # Extract document texts
            documents = []
            for candidate in candidates[:self.rerank_k]:  # Only rerank top rerank_k
                doc_text = candidate.get("text", "")
                if not doc_text:
                    # Fallback to title if text is missing
                    doc_text = candidate.get("title", "")
                documents.append(doc_text)
            
            if not documents:
                logger.warning("No documents to rerank")
                return candidates
            
            # Re-rank
            rerank_results = self.reranker.rerank(
                query=query_text,
                documents=documents,
                top_k=len(documents)
            )
            
            # Map rerank scores back to candidates
            rerank_score_map = {r["index"]: r["rerank_score"] for r in rerank_results}
            
            # Add rerank scores to candidates
            for i, candidate in enumerate(candidates[:self.rerank_k]):
                candidate["rerank_score"] = rerank_score_map.get(i, 0.0)
            
            # Sort by rerank_score and return top rerank_k
            reranked = sorted(
                candidates[:self.rerank_k],
                key=lambda x: x.get("rerank_score", 0.0),
                reverse=True
            )
            
            logger.info(f"Stage 2: Re-ranked {len(reranked)} candidates")
            return reranked
            
        except (RerankerAPIError, RerankerModelError) as e:
            logger.warning(f"Stage 2 reranking failed: {e}. Continuing with original scores.")
            # Add default rerank_score of 0.0
            for candidate in candidates[:self.rerank_k]:
                candidate["rerank_score"] = 0.0
            return candidates[:self.rerank_k]
        except Exception as e:
            logger.error(f"Stage 2 unexpected error: {e}", exc_info=True)
            # Add default rerank_score of 0.0
            for candidate in candidates[:self.rerank_k]:
                candidate["rerank_score"] = 0.0
            return candidates[:self.rerank_k]
    
    def _stage3_kg_scoring(
        self,
        candidates: List[Dict[str, Any]],
        fault_codes: List[str]
    ) -> Dict[str, float]:
        """
        Stage 3: Knowledge graph path scoring.
        
        Args:
            candidates: List of candidates from Stage 2
            fault_codes: List of fault code strings
        
        Returns:
            Dictionary mapping procedure_id -> max path score
        """
        if self.kg_query is None:
            logger.debug("Knowledge graph disabled, skipping Stage 3")
            return {}
        
        kg_scores: Dict[str, float] = {}
        
        try:
            # Get KG config
            kg_config = self.config.get("knowledge_graph", {})
            max_path_length = kg_config.get("max_path_length", 3)
            
            # For each fault code, find paths to procedures
            for fault_code in fault_codes:
                try:
                    # Get procedures for this fault code
                    procedures = self.kg_query.get_procedures_for_fault(
                        fault_code=fault_code,
                        max_length=max_path_length
                    )
                    
                    # Update scores (use max score if multiple paths exist)
                    for proc_info in procedures:
                        procedure_id = proc_info.get("procedure_id", "")
                        path_score = proc_info.get("path_score", 0.0)
                        
                        if procedure_id:
                            # Use maximum score if multiple fault codes point to same procedure
                            if procedure_id in kg_scores:
                                kg_scores[procedure_id] = max(kg_scores[procedure_id], path_score)
                            else:
                                kg_scores[procedure_id] = path_score
                
                except Exception as e:
                    logger.debug(f"Error processing fault code {fault_code} in KG: {e}")
                    continue
            
            logger.info(f"Stage 3: Computed KG scores for {len(kg_scores)} procedures")
            return kg_scores
            
        except Exception as e:
            logger.warning(f"Stage 3 KG scoring failed: {e}. Continuing without KG scores.")
            return {}
    
    def _stage4_combined_scoring(
        self,
        candidates: List[Dict[str, Any]],
        kg_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Stage 4: Combined scoring using Ranker.
        
        Args:
            candidates: List of candidates from Stage 2
            kg_scores: Dictionary mapping procedure_id -> KG path score
        
        Returns:
            List of candidates ranked by combined_score
        """
        try:
            # Get feedback scores for procedures
            feedback_scores: Dict[str, float] = {}
            if self.feedback_collector is not None:
                for candidate in candidates:
                    procedure_id = candidate.get("procedure_id", "")
                    if procedure_id:
                        try:
                            score = self.feedback_collector.get_procedure_score(procedure_id)
                            if score is not None:
                                feedback_scores[procedure_id] = score
                        except Exception as e:
                            logger.debug(f"Error getting feedback score for {procedure_id}: {e}")
                            continue
            
            # Use Ranker to combine scores
            ranked = self.ranker.rank(
                candidates=candidates,
                kg_scores=kg_scores,
                feedback_scores=feedback_scores if feedback_scores else None
            )
            
            logger.info(f"Stage 4: Ranked {len(ranked)} candidates")
            return ranked
            
        except RankerError as e:
            logger.error(f"Stage 4 ranking failed: {e}. Returning candidates without combined scores.")
            # Return candidates sorted by original score
            return sorted(
                candidates,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Stage 4 unexpected error: {e}", exc_info=True)
            # Return candidates sorted by original score
            return sorted(
                candidates,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )

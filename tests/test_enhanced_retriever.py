"""
Tests for EnhancedRetriever multi-stage retrieval pipeline.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import yaml
import tempfile
import os

from src.retrieval.enhanced_retriever import EnhancedRetriever, EnhancedRetrieverError
from src.retrieval.vector_store import VectorStoreOperationError
from src.retrieval.reranker import RerankerAPIError, RerankerModelError
from src.retrieval.ranker import RankerError


@pytest.fixture
def mock_config():
    """Create a temporary config file for testing."""
    config = {
        "vector_store": {
            "provider": "qdrant",
            "collection_name": "test_collection",
            "distance_metric": "cosine",
            "vector_size": 768,
            "url": "http://localhost:6333"
        },
        "retrieval": {
            "initial_k": 100,
            "rerank_k": 50,
            "final_k": 10
        },
        "reranking": {
            "enabled": True,
            "provider": "local",
            "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "top_k": 50
        },
        "ranking": {
            "embedding_similarity": 0.4,
            "rerank_score": 0.3,
            "kg_path_score": 0.2,
            "feedback_score": 0.1
        },
        "knowledge_graph": {
            "enabled": True,
            "graph_path": "data/knowledge_graph.graphml",
            "max_path_length": 3
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    # Mock VectorStore
    vector_store = Mock()
    vector_store.search.return_value = [
        {
            "id": "doc1",
            "score": 0.9,
            "text": "Test document 1",
            "title": "Title 1",
            "procedure_id": "proc1"
        },
        {
            "id": "doc2",
            "score": 0.8,
            "text": "Test document 2",
            "title": "Title 2",
            "procedure_id": "proc2"
        }
    ]
    
    # Mock Reranker
    reranker = Mock()
    reranker.enabled = True
    reranker.rerank.return_value = [
        {"index": 0, "rerank_score": 0.85},
        {"index": 1, "rerank_score": 0.75}
    ]
    
    # Mock KnowledgeGraphQuery
    kg_query = Mock()
    kg_query.get_procedures_for_fault.return_value = [
        {
            "procedure_id": "proc1",
            "path_score": 0.9,
            "path": ["fault_code:P0301", "procedure:proc1"]
        }
    ]
    
    # Mock Ranker
    ranker = Mock()
    ranker.rank.return_value = [
        {
            "id": "doc1",
            "score": 0.9,
            "rerank_score": 0.85,
            "procedure_id": "proc1",
            "combined_score": 0.88,
            "text": "Test document 1",
            "title": "Title 1"
        },
        {
            "id": "doc2",
            "score": 0.8,
            "rerank_score": 0.75,
            "procedure_id": "proc2",
            "combined_score": 0.78,
            "text": "Test document 2",
            "title": "Title 2"
        }
    ]
    
    # Mock FeedbackCollector
    feedback_collector = Mock()
    feedback_collector.get_procedure_score.return_value = 0.7
    
    # Mock MultiModalEncoder
    encoder = Mock()
    encoder.eval = Mock()
    mock_embedding = torch.tensor([[0.1] * 768], dtype=torch.float32)
    encoder.encode.return_value = mock_embedding
    
    return {
        "vector_store": vector_store,
        "reranker": reranker,
        "kg_query": kg_query,
        "ranker": ranker,
        "feedback_collector": feedback_collector,
        "encoder": encoder
    }


def test_enhanced_retriever_init(mock_config):
    """Test EnhancedRetriever initialization."""
    with patch('src.retrieval.enhanced_retriever.VectorStore') as mock_vs, \
         patch('src.retrieval.enhanced_retriever.Reranker') as mock_reranker, \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery') as mock_kg, \
         patch('src.retrieval.enhanced_retriever.Ranker') as mock_ranker, \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector') as mock_fb, \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder') as mock_encoder:
        
        mock_vs.return_value = Mock()
        mock_reranker.return_value = Mock(enabled=True)
        mock_kg.return_value = Mock()
        mock_ranker.return_value = Mock()
        mock_fb.return_value = Mock()
        mock_encoder.return_value = Mock()
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        assert retriever.initial_k == 100
        assert retriever.rerank_k == 50
        assert retriever.final_k == 10
        assert retriever.vector_store is not None
        assert retriever.reranker is not None


def test_enhanced_retriever_init_missing_config():
    """Test EnhancedRetriever initialization with missing config file."""
    with pytest.raises(EnhancedRetrieverError, match="Config file not found"):
        EnhancedRetriever(config_path="/nonexistent/config.yaml")


def test_stage1_vector_search(mock_config, mock_components):
    """Test Stage 1: Vector search."""
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        candidates = retriever._stage1_vector_search(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000}
        )
        
        assert len(candidates) == 2
        assert candidates[0]["id"] == "doc1"
        mock_components["vector_store"].search.assert_called_once()


def test_stage1_vector_search_error(mock_config, mock_components):
    """Test Stage 1 error handling."""
    mock_components["vector_store"].search.side_effect = VectorStoreOperationError("Search failed")
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        candidates = retriever._stage1_vector_search(
            fault_codes=["P0301"],
            obd_data=None
        )
        
        assert len(candidates) == 0


def test_stage2_reranking(mock_config, mock_components):
    """Test Stage 2: Re-ranking."""
    candidates = [
        {"id": "doc1", "score": 0.9, "text": "Test 1", "procedure_id": "proc1"},
        {"id": "doc2", "score": 0.8, "text": "Test 2", "procedure_id": "proc2"}
    ]
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        reranked = retriever._stage2_reranking(
            candidates=candidates,
            fault_codes=["P0301"],
            query_text="Test query"
        )
        
        assert len(reranked) <= 50  # Should respect rerank_k
        assert "rerank_score" in reranked[0]
        mock_components["reranker"].rerank.assert_called_once()


def test_stage2_reranking_disabled(mock_config, mock_components):
    """Test Stage 2 when reranker is disabled."""
    mock_components["reranker"].enabled = False
    candidates = [
        {"id": "doc1", "score": 0.9, "text": "Test 1", "procedure_id": "proc1"}
    ]
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        reranked = retriever._stage2_reranking(
            candidates=candidates,
            fault_codes=["P0301"],
            query_text=None
        )
        
        assert len(reranked) <= 50
        assert reranked[0]["rerank_score"] == 0.0
        mock_components["reranker"].rerank.assert_not_called()


def test_stage2_reranking_error(mock_config, mock_components):
    """Test Stage 2 error handling."""
    mock_components["reranker"].rerank.side_effect = RerankerAPIError("API error")
    candidates = [
        {"id": "doc1", "score": 0.9, "text": "Test 1", "procedure_id": "proc1"}
    ]
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        reranked = retriever._stage2_reranking(
            candidates=candidates,
            fault_codes=["P0301"],
            query_text=None
        )
        
        assert len(reranked) <= 50
        assert reranked[0]["rerank_score"] == 0.0  # Default score on error


def test_stage3_kg_scoring(mock_config, mock_components):
    """Test Stage 3: KG path scoring."""
    candidates = [
        {"id": "doc1", "score": 0.9, "procedure_id": "proc1"},
        {"id": "doc2", "score": 0.8, "procedure_id": "proc2"}
    ]
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        kg_scores = retriever._stage3_kg_scoring(
            candidates=candidates,
            fault_codes=["P0301"]
        )
        
        assert isinstance(kg_scores, dict)
        assert "proc1" in kg_scores
        mock_components["kg_query"].get_procedures_for_fault.assert_called()


def test_stage3_kg_scoring_disabled(mock_config, mock_components):
    """Test Stage 3 when KG is disabled."""
    candidates = [
        {"id": "doc1", "score": 0.9, "procedure_id": "proc1"}
    ]
    
    # Create config with KG disabled
    config = {
        "vector_store": {"provider": "qdrant", "collection_name": "test", "vector_size": 768, "url": "http://localhost:6333"},
        "retrieval": {"initial_k": 100, "rerank_k": 50, "final_k": 10},
        "reranking": {"enabled": True, "provider": "local", "model": "test", "top_k": 50},
        "ranking": {"embedding_similarity": 0.4, "rerank_score": 0.3, "kg_path_score": 0.2, "feedback_score": 0.1},
        "knowledge_graph": {"enabled": False}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
             patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
             patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
             patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
             patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
            
            retriever = EnhancedRetriever(config_path=config_path)
            
            kg_scores = retriever._stage3_kg_scoring(
                candidates=candidates,
                fault_codes=["P0301"]
            )
            
            assert kg_scores == {}
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_stage4_combined_scoring(mock_config, mock_components):
    """Test Stage 4: Combined scoring."""
    candidates = [
        {"id": "doc1", "score": 0.9, "rerank_score": 0.85, "procedure_id": "proc1"},
        {"id": "doc2", "score": 0.8, "rerank_score": 0.75, "procedure_id": "proc2"}
    ]
    kg_scores = {"proc1": 0.9, "proc2": 0.7}
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        ranked = retriever._stage4_combined_scoring(
            candidates=candidates,
            kg_scores=kg_scores
        )
        
        assert len(ranked) == 2
        assert "combined_score" in ranked[0]
        mock_components["ranker"].rank.assert_called_once()


def test_stage4_combined_scoring_error(mock_config, mock_components):
    """Test Stage 4 error handling."""
    mock_components["ranker"].rank.side_effect = RankerError("Ranking failed")
    candidates = [
        {"id": "doc1", "score": 0.9, "rerank_score": 0.85, "procedure_id": "proc1"}
    ]
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        ranked = retriever._stage4_combined_scoring(
            candidates=candidates,
            kg_scores={}
        )
        
        # Should fallback to sorting by original score
        assert len(ranked) == 1
        assert ranked[0]["id"] == "doc1"


def test_full_retrieval_pipeline(mock_config, mock_components):
    """Test full retrieval pipeline end-to-end."""
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        results = retriever.retrieve(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000},
            query_text="Test query",
            top_k=5
        )
        
        assert len(results) <= 5
        assert len(results) > 0
        assert "combined_score" in results[0]
        
        # Verify all stages were called
        mock_components["vector_store"].search.assert_called_once()
        mock_components["reranker"].rerank.assert_called_once()
        mock_components["kg_query"].get_procedures_for_fault.assert_called()
        mock_components["ranker"].rank.assert_called_once()


def test_retrieve_empty_fault_codes(mock_config, mock_components):
    """Test retrieve with empty fault codes."""
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        results = retriever.retrieve(fault_codes=[])
        
        assert len(results) == 0


def test_retrieve_no_candidates_stage1(mock_config, mock_components):
    """Test retrieve when Stage 1 returns no candidates."""
    mock_components["vector_store"].search.return_value = []
    
    with patch('src.retrieval.enhanced_retriever.VectorStore', return_value=mock_components["vector_store"]), \
         patch('src.retrieval.enhanced_retriever.Reranker', return_value=mock_components["reranker"]), \
         patch('src.retrieval.enhanced_retriever.KnowledgeGraphQuery', return_value=mock_components["kg_query"]), \
         patch('src.retrieval.enhanced_retriever.Ranker', return_value=mock_components["ranker"]), \
         patch('src.retrieval.enhanced_retriever.FeedbackCollector', return_value=mock_components["feedback_collector"]), \
         patch('src.retrieval.enhanced_retriever.MultiModalEncoder', return_value=mock_components["encoder"]):
        
        retriever = EnhancedRetriever(config_path=mock_config)
        
        results = retriever.retrieve(fault_codes=["P0301"])
        
        assert len(results) == 0

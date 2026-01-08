"""
Tests for retrieval modules.
"""
import pytest
import numpy as np
from src.retrieval.ranker import Ranker


def test_ranker():
    """Test ranker combined scoring"""
    config = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1
    }
    ranker = Ranker(config)
    
    candidates = [
        {"score": 0.8, "rerank_score": 0.7, "procedure_id": "proc1"},
        {"score": 0.6, "rerank_score": 0.5, "procedure_id": "proc2"}
    ]
    
    ranked = ranker.rank(candidates)
    
    assert len(ranked) == 2
    assert "combined_score" in ranked[0]
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]

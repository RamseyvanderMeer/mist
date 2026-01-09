"""
Tests for retrieval modules.
"""
import pytest
import numpy as np
from pathlib import Path
from src.retrieval.ranker import Ranker, RankerConfigurationError


def test_ranker_basic():
    """Test ranker combined scoring with all scores"""
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
    kg_scores = {"proc1": 0.9, "proc2": 0.3}
    feedback_scores = {"proc1": 0.8, "proc2": 0.4}
    
    ranked = ranker.rank(candidates, kg_scores=kg_scores, feedback_scores=feedback_scores)
    
    assert len(ranked) == 2
    assert "combined_score" in ranked[0]
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]
    # proc1 should rank higher due to better scores
    assert ranked[0]["procedure_id"] == "proc1"


def test_ranker_missing_scores():
    """Test ranker handles missing scores gracefully"""
    config = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1
    }
    ranker = Ranker(config)
    
    candidates = [
        {"score": 0.8, "procedure_id": "proc1"},  # Missing rerank_score
        {"rerank_score": 0.5, "procedure_id": "proc2"}  # Missing score
    ]
    
    ranked = ranker.rank(candidates)
    
    assert len(ranked) == 2
    assert "combined_score" in ranked[0]
    # proc1 should rank higher (has embedding score)
    assert ranked[0]["procedure_id"] == "proc1"
    # Missing scores should default correctly
    assert ranked[0].get("rerank_score", 0.0) == 0.0  # Default for missing


def test_ranker_missing_kg_feedback():
    """Test ranker with missing KG and feedback scores"""
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
    
    # No KG or feedback scores provided
    ranked = ranker.rank(candidates)
    
    assert len(ranked) == 2
    assert "combined_score" in ranked[0]
    # Should still rank correctly based on embedding and rerank scores
    assert ranked[0]["procedure_id"] == "proc1"


def test_ranker_score_normalization():
    """Test ranker normalizes scores outside [0, 1] range"""
    config = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1
    }
    ranker = Ranker(config)
    
    candidates = [
        {"score": 1.5, "rerank_score": -0.2, "procedure_id": "proc1"},  # Out of range
        {"score": 0.9, "rerank_score": 0.9, "procedure_id": "proc2"}  # Normal, high scores
    ]
    
    ranked = ranker.rank(candidates)
    
    assert len(ranked) == 2
    # After clamping proc1: score=1.0, rerank=0.0 -> combined = 0.4*1.0 + 0.3*0.0 = 0.4
    # proc2: score=0.9, rerank=0.9 -> combined = 0.4*0.9 + 0.3*0.9 = 0.63
    # So proc2 should rank higher
    assert ranked[0]["procedure_id"] == "proc2"
    # Combined scores should be in valid range
    assert 0.0 <= ranked[0]["combined_score"] <= 1.0
    assert 0.0 <= ranked[1]["combined_score"] <= 1.0


def test_ranker_weight_normalization():
    """Test ranker normalizes weights that don't sum to 1.0"""
    config = {
        "embedding_similarity": 0.8,  # Sums to 2.0, should normalize
        "rerank_score": 0.6,
        "kg_path_score": 0.4,
        "feedback_score": 0.2
    }
    ranker = Ranker(config)
    
    # Weights should be normalized to sum to 1.0
    total = sum(ranker.weights.values())
    assert abs(total - 1.0) < 0.001
    
    candidates = [
        {"score": 0.8, "rerank_score": 0.7, "procedure_id": "proc1"},
        {"score": 0.6, "rerank_score": 0.5, "procedure_id": "proc2"}
    ]
    
    ranked = ranker.rank(candidates)
    assert len(ranked) == 2


def test_ranker_empty_candidates():
    """Test ranker handles empty candidates list"""
    config = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1
    }
    ranker = Ranker(config)
    
    ranked = ranker.rank([])
    assert len(ranked) == 0


def test_ranker_default_config():
    """Test ranker loads from default config file"""
    # Should load from retrieval_config.yaml
    ranker = Ranker(config=None)
    
    # Check that weights are set (from config file)
    assert "embedding_similarity" in ranker.weights
    assert "rerank_score" in ranker.weights
    assert "kg_path_score" in ranker.weights
    assert "feedback_score" in ranker.weights
    
    # Weights should sum to 1.0
    total = sum(ranker.weights.values())
    assert abs(total - 1.0) < 0.001


def test_ranker_invalid_weights():
    """Test ranker raises error for invalid weights"""
    # Negative weight should raise error
    with pytest.raises(RankerConfigurationError):
        Ranker(config={"embedding_similarity": -0.4})
    
    # All zeros should raise error
    with pytest.raises(RankerConfigurationError):
        Ranker(config={
            "embedding_similarity": 0.0,
            "rerank_score": 0.0,
            "kg_path_score": 0.0,
            "feedback_score": 0.0
        })


def test_ranker_feedback_default_neutral():
    """Test ranker uses 0.5 as default for missing feedback scores"""
    config = {
        "embedding_similarity": 0.0,
        "rerank_score": 0.0,
        "kg_path_score": 0.0,
        "feedback_score": 1.0  # Only feedback matters
    }
    ranker = Ranker(config)
    
    candidates = [
        {"score": 0.8, "rerank_score": 0.7, "procedure_id": "proc1"},
        {"score": 0.6, "rerank_score": 0.5, "procedure_id": "proc2"}
    ]
    
    # No feedback scores provided - should use default 0.5 for both
    ranked = ranker.rank(candidates)
    
    assert len(ranked) == 2
    # Both should have same combined score (0.5) since only feedback matters
    assert abs(ranked[0]["combined_score"] - ranked[1]["combined_score"]) < 0.001


def test_ranker_ranking_order():
    """Test ranker correctly orders candidates by combined score"""
    config = {
        "embedding_similarity": 0.4,
        "rerank_score": 0.3,
        "kg_path_score": 0.2,
        "feedback_score": 0.1
    }
    ranker = Ranker(config)
    
    candidates = [
        {"score": 0.5, "rerank_score": 0.5, "procedure_id": "proc1"},
        {"score": 0.9, "rerank_score": 0.9, "procedure_id": "proc2"},
        {"score": 0.3, "rerank_score": 0.3, "procedure_id": "proc3"},
    ]
    kg_scores = {"proc1": 0.5, "proc2": 0.9, "proc3": 0.3}
    feedback_scores = {"proc1": 0.5, "proc2": 0.9, "proc3": 0.3}
    
    ranked = ranker.rank(candidates, kg_scores=kg_scores, feedback_scores=feedback_scores)
    
    assert len(ranked) == 3
    # Should be ordered: proc2 > proc1 > proc3
    assert ranked[0]["procedure_id"] == "proc2"
    assert ranked[1]["procedure_id"] == "proc1"
    assert ranked[2]["procedure_id"] == "proc3"
    
    # Scores should be descending
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]
    assert ranked[1]["combined_score"] >= ranked[2]["combined_score"]
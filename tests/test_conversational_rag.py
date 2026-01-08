"""
Tests for conversational RAG.
"""
import pytest
from src.retrieval.conversational_rag import ConversationalRAG


def test_conversational_rag_init():
    """Test ConversationalRAG initialization"""
    rag = ConversationalRAG()
    assert rag is not None


def test_query_structure():
    """Test query response structure"""
    rag = ConversationalRAG()
    response = rag.query(
        fault_codes=["P0300"],
        obd_data={"engine_rpm": 2500}
    )
    
    assert "recommendations" in response
    assert "needs_clarification" in response
    assert "session_id" in response

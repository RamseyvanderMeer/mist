"""
Unit tests for FastAPI server endpoints.

Tests each endpoint with mock dependencies and error handling.
"""
import pytest
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

from src.api.server import app, get_conversational_rag, get_feedback_collector, get_feedback_analyzer
from src.api.schemas import (
    QueryRequest, QueryResponse, ClarifyRequest, RatingFeedback,
    RepairOutcomeFeedback, ConversationCorrection, FeedbackStatistics
)
from src.retrieval.conversational_rag import ConversationalRAGError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_rag():
    """Fixture for ConversationalRAG mock."""
    return Mock()


@pytest.fixture
def mock_collector():
    """Fixture for FeedbackCollector mock."""
    return Mock()


@pytest.fixture
def mock_analyzer():
    """Fixture for FeedbackAnalyzer mock."""
    return Mock()


@pytest.fixture
def client():
    """Fixture for TestClient with clean dependency state."""
    # Ensure clean state before each test
    app.dependency_overrides.clear()
    yield TestClient(app)
    # Clean up after each test
    app.dependency_overrides.clear()


@pytest.fixture
def override_rag(mock_rag):
    """Fixture to override ConversationalRAG dependency."""
    app.dependency_overrides[get_conversational_rag] = lambda: mock_rag
    yield mock_rag
    app.dependency_overrides.pop(get_conversational_rag, None)


@pytest.fixture
def override_collector(mock_collector):
    """Fixture to override FeedbackCollector dependency."""
    app.dependency_overrides[get_feedback_collector] = lambda: mock_collector
    yield mock_collector
    app.dependency_overrides.pop(get_feedback_collector, None)


@pytest.fixture
def override_analyzer(mock_analyzer):
    """Fixture to override FeedbackAnalyzer dependency."""
    app.dependency_overrides[get_feedback_analyzer] = lambda: mock_analyzer
    yield mock_analyzer
    app.dependency_overrides.pop(get_feedback_analyzer, None)


# ============================================================================
# Helper Functions
# ============================================================================

def create_query_response(session_id="session_123", recommendations=None, needs_clarification=False):
    """Helper to create a query response dict."""
    if recommendations is None:
        recommendations = []
    return {
        "recommendations": recommendations,
        "needs_clarification": needs_clarification,
        "clarification_questions": None if not needs_clarification else ["Question 1"],
        "session_id": session_id,
        "query_text": "P0301, P0302"
    }


# ============================================================================
# Test Classes
# ============================================================================

class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "MIST API"}


class TestQueryEndpoint:
    """Test /query endpoint."""
    
    def test_query_success(self, client, override_rag):
        """Test successful query processing."""
        override_rag.query.return_value = create_query_response(
            recommendations=[{
                "id": "rec_1",
                "title": "Test Guide",
                "procedure_name": "Test Procedure",
                "procedure_id": "proc_123",
                "score": 0.85,
                "text": "Some text"
            }]
        )
        
        request_data = {
            "fault_codes": ["P0301", "P0302"],
            "obd_data": {"rpm": 2000, "temp": 90}
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_123"
        assert len(data["recommendations"]) == 1
        assert data["needs_clarification"] is False
        override_rag.query.assert_called_once()
    
    def test_query_with_vehicle_context(self, client, override_rag):
        """Test query with vehicle context."""
        override_rag.query.return_value = create_query_response(session_id="session_456")
        
        request_data = {
            "fault_codes": ["P0301"],
            "obd_data": {"rpm": 2000},
            "vehicle_context": {"model": "BMW 320i", "year": 2020}
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 200
        call_args = override_rag.query.call_args
        assert call_args.kwargs["vehicle_context"]["model"] == "BMW 320i"
    
    def test_query_with_session_id(self, client, override_rag):
        """Test query with existing session_id."""
        override_rag.query.return_value = create_query_response(session_id="existing_session")
        
        request_data = {
            "fault_codes": ["P0301"],
            "obd_data": {},
            "session_id": "existing_session"
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 200
        call_args = override_rag.query.call_args
        assert call_args.kwargs["session_id"] == "existing_session"
    
    def test_query_rag_error(self, client, override_rag):
        """Test query endpoint handles ConversationalRAGError."""
        override_rag.query.side_effect = ConversationalRAGError("RAG processing failed")
        
        request_data = {
            "fault_codes": ["P0301"],
            "obd_data": {}
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]
    
    def test_query_validation_error(self, client, override_rag):
        """Test query endpoint handles validation errors."""
        # Override to prevent initialization, but validation happens before dependency is called
        override_rag.query.return_value = create_query_response()
        
        # Missing required field
        request_data = {
            "fault_codes": ["P0301"]
            # Missing obd_data
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_empty_fault_codes(self, client, override_rag):
        """Test query endpoint rejects empty fault codes."""
        # Override to prevent initialization
        override_rag.query.return_value = create_query_response()
        
        request_data = {
            "fault_codes": [],
            "obd_data": {}
        }
        response = client.post("/query", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestClarifyEndpoint:
    """Test /clarify endpoint."""
    
    def test_clarify_success(self, client, override_rag):
        """Test successful clarification processing."""
        override_rag.clarify.return_value = {
            "recommendations": [{
                "id": "rec_1",
                "title": "Refined Guide",
                "procedure_name": "Refined Procedure",
                "score": 0.95
            }],
            "needs_clarification": False,
            "clarification_questions": None,
            "session_id": "session_123",
            "query_text": "Expanded query"
        }
        
        request_data = {
            "session_id": "session_123",
            "responses": ["Answer 1", "Answer 2"]
        }
        response = client.post("/clarify", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_123"
        assert len(data["recommendations"]) == 1
        assert data["needs_clarification"] is False
        override_rag.clarify.assert_called_once_with(
            session_id="session_123",
            responses=["Answer 1", "Answer 2"]
        )
    
    def test_clarify_session_not_found(self, client, override_rag):
        """Test clarify endpoint handles session not found."""
        override_rag.clarify.side_effect = ConversationalRAGError("Session session_999 not found")
        
        request_data = {
            "session_id": "session_999",
            "responses": ["Answer 1"]
        }
        response = client.post("/clarify", json=request_data)
        
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
    
    def test_clarify_rag_error(self, client, override_rag):
        """Test clarify endpoint handles other RAG errors."""
        override_rag.clarify.side_effect = ConversationalRAGError("Processing error")
        
        request_data = {
            "session_id": "session_123",
            "responses": ["Answer 1"]
        }
        response = client.post("/clarify", json=request_data)
        
        assert response.status_code == 500
        assert "Clarification processing failed" in response.json()["detail"]
    
    def test_clarify_validation_error(self, client, override_rag):
        """Test clarify endpoint handles validation errors."""
        # Override to prevent initialization
        override_rag.clarify.return_value = create_query_response()
        
        # Missing required field
        request_data = {
            "session_id": "session_123"
            # Missing responses
        }
        response = client.post("/clarify", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestFeedbackRatingEndpoint:
    """Test /feedback/rating endpoint."""
    
    def test_submit_rating_success(self, client, override_collector):
        """Test successful rating submission."""
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "rating": 5,
            "selected_guide": "guide_456"
        }
        response = client.post("/feedback/rating", json=request_data)
        
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "Rating recorded"}
        override_collector.save_session.assert_called_once()
    
    def test_submit_rating_without_selected_guide(self, client, override_collector):
        """Test rating submission without selected guide."""
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "rating": 4
        }
        response = client.post("/feedback/rating", json=request_data)
        
        assert response.status_code == 200
    
    def test_submit_rating_invalid_rating(self, client, override_collector):
        """Test rating submission with invalid rating value."""
        # Note: Rating validation happens at Pydantic level, so ValueError from save_session
        # would only occur if there's a bug. But we test the error handling path.
        override_collector.save_session.side_effect = ValueError("Rating must be between 1 and 5")
        
        request_data = {
            "session_id": "session_123",
            "rating": 5  # Valid at Pydantic level, but mock raises ValueError
        }
        response = client.post("/feedback/rating", json=request_data)
        
        assert response.status_code == 400
    
    def test_submit_rating_session_not_found(self, client, override_collector):
        """Test rating submission with non-existent session."""
        override_collector.save_session.side_effect = RuntimeError("Session does not exist")
        
        request_data = {
            "session_id": "nonexistent",
            "rating": 5
        }
        response = client.post("/feedback/rating", json=request_data)
        
        # RuntimeError with "does not exist" should return 404
        assert response.status_code == 404
    
    def test_submit_rating_validation_error(self, client):
        """Test rating submission with validation error."""
        # Invalid rating value - caught by Pydantic
        request_data = {
            "session_id": "session_123",
            "rating": 0  # Below minimum
        }
        response = client.post("/feedback/rating", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestFeedbackOutcomeEndpoint:
    """Test /feedback/outcome endpoint."""
    
    def test_submit_outcome_success(self, client, override_collector):
        """Test successful outcome submission."""
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "outcome": "success"
        }
        response = client.post("/feedback/outcome", json=request_data)
        
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "Outcome recorded"}
    
    def test_submit_outcome_with_details(self, client, override_collector):
        """Test outcome submission with details."""
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "outcome": "failure",
            "details": {"notes": "Part not available"}
        }
        response = client.post("/feedback/outcome", json=request_data)
        
        assert response.status_code == 200
    
    def test_submit_outcome_invalid_outcome(self, client):
        """Test outcome submission with invalid outcome."""
        # Invalid outcome is caught by Pydantic validation (422)
        request_data = {
            "session_id": "session_123",
            "outcome": "invalid"
        }
        response = client.post("/feedback/outcome", json=request_data)
        
        assert response.status_code == 422  # Validation error, not 400


class TestFeedbackCorrectionEndpoint:
    """Test /feedback/correction endpoint."""
    
    def test_submit_correction_success(self, client, override_collector):
        """Test successful correction submission."""
        override_collector.get_session.return_value = {
            "session_id": "session_123",
            "conversation_corrections": []
        }
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "correction": {"field": "question", "value": "Corrected"}
        }
        response = client.post("/feedback/correction", json=request_data)
        
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "Correction recorded"}
    
    def test_submit_correction_new_session(self, client, override_collector):
        """Test correction submission for new session."""
        override_collector.get_session.return_value = None  # New session
        override_collector.save_session.return_value = "session_123"
        
        request_data = {
            "session_id": "session_123",
            "correction": {"field": "question", "value": "Corrected"}
        }
        response = client.post("/feedback/correction", json=request_data)
        
        assert response.status_code == 200


class TestFeedbackGetSessionEndpoint:
    """Test /feedback/{session_id} GET endpoint."""
    
    def test_get_session_success(self, client, override_collector):
        """Test successful session retrieval."""
        override_collector.get_session.return_value = {
            "session_id": "session_123",
            "fault_codes": ["P0301"],
            "explicit_rating": 5
        }
        
        response = client.get("/feedback/session_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_123"
        assert data["explicit_rating"] == 5
    
    def test_get_session_not_found(self, client, override_collector):
        """Test session retrieval for non-existent session."""
        override_collector.get_session.return_value = None
        
        response = client.get("/feedback/nonexistent")
        
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]


class TestFeedbackStatisticsEndpoint:
    """Test /feedback/statistics GET endpoint."""
    
    def test_get_statistics_success(self, client, override_analyzer, override_collector):
        """Test successful statistics retrieval."""
        # Mock analyzer statistics
        override_analyzer.get_statistics.return_value = {
            "total_sessions": 100,
            "rated_sessions": 80,
            "average_rating": 4.2,
            "repair_outcomes": {"success": 50, "failure": 20, "partial": 10},
            "rating_coverage": 0.8
        }
        
        # Mock database connection for corrected_sessions
        # The imports are done inside the endpoint function, so we patch the module
        from unittest.mock import patch
        
        mock_session = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        mock_filter.count.return_value = 5
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        
        mock_connection = Mock()
        mock_connection.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_connection.session.return_value.__exit__ = Mock(return_value=None)
        
        mock_paths = Mock()
        mock_paths.feedback_db = "/tmp/test.db"
        
        # Patch the imports that are done inside the endpoint
        with patch('src.database.connection.create_connection', return_value=mock_connection), \
             patch('src.paths.get_paths', return_value=mock_paths):
            response = client.get("/feedback/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_sessions"] == 100
        assert data["rated_sessions"] == 80
        assert data["average_rating"] == 4.2
        assert data["corrected_sessions"] == 5
        assert data["rating_coverage"] == 0.8
    
    def test_get_statistics_analyzer_error(self, client, override_analyzer):
        """Test statistics endpoint handles analyzer errors."""
        from sqlalchemy.exc import SQLAlchemyError
        
        override_analyzer.get_statistics.side_effect = SQLAlchemyError("Database error")
        
        response = client.get("/feedback/statistics")
        
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

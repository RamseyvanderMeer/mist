"""
Unit tests for API schemas (Pydantic models).

Tests model validation, serialization/deserialization, and edge cases.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas import (
    Recommendation,
    QueryRequest,
    QueryResponse,
    ClarifyRequest,
    RepairOutcome,
    RatingFeedback,
    RepairOutcomeFeedback,
    ConversationCorrection,
    FeedbackStatistics,
)


class TestRecommendation:
    """Test Recommendation model."""
    
    def test_valid_recommendation(self):
        """Test creating a valid recommendation."""
        rec = Recommendation(
            id="rec_1",
            title="Test Repair Guide",
            procedure_name="Test Procedure",
            procedure_id="proc_123",
            score=0.85,
            text="Some repair text"
        )
        assert rec.id == "rec_1"
        assert rec.score == 0.85
        assert rec.procedure_id == "proc_123"
    
    def test_recommendation_with_optional_fields(self):
        """Test recommendation with optional fields as None."""
        rec = Recommendation(
            id="rec_2",
            title="Test Guide",
            procedure_name="Test",
            score=0.5
        )
        assert rec.procedure_id is None
        assert rec.text is None
    
    def test_recommendation_score_validation_min(self):
        """Test that score must be >= 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            Recommendation(
                id="rec_3",
                title="Test",
                procedure_name="Test",
                score=-0.1
            )
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_recommendation_score_validation_max(self):
        """Test that score must be <= 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Recommendation(
                id="rec_4",
                title="Test",
                procedure_name="Test",
                score=1.1
            )
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_recommendation_serialization(self):
        """Test serialization to dict."""
        rec = Recommendation(
            id="rec_5",
            title="Test",
            procedure_name="Test",
            score=0.75
        )
        data = rec.model_dump()
        assert data["id"] == "rec_5"
        assert data["score"] == 0.75
        assert isinstance(data, dict)
    
    def test_recommendation_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "id": "rec_6",
            "title": "Test",
            "procedure_name": "Test",
            "score": 0.9
        }
        rec = Recommendation(**data)
        assert rec.id == "rec_6"
        assert rec.score == 0.9


class TestQueryRequest:
    """Test QueryRequest model."""
    
    def test_valid_query_request(self):
        """Test creating a valid query request."""
        request = QueryRequest(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90}
        )
        assert len(request.fault_codes) == 2
        assert request.vehicle_context is None
        assert request.session_id is None
    
    def test_query_request_with_vehicle_context(self):
        """Test query request with vehicle context."""
        request = QueryRequest(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000},
            vehicle_context={"model": "BMW 320i", "year": 2020}
        )
        assert request.vehicle_context["model"] == "BMW 320i"
        assert request.vehicle_context["year"] == 2020
    
    def test_query_request_with_session_id(self):
        """Test query request with session ID."""
        request = QueryRequest(
            fault_codes=["P0301"],
            obd_data={},
            session_id="session_123"
        )
        assert request.session_id == "session_123"
    
    def test_query_request_empty_fault_codes_validation(self):
        """Test that fault_codes cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                fault_codes=[],
                obd_data={}
            )
        assert "at least 1 item" in str(exc_info.value)
    
    def test_query_request_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            QueryRequest(fault_codes=["P0301"])  # Missing obd_data
    
    def test_query_request_serialization(self):
        """Test serialization to dict."""
        request = QueryRequest(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000},
            vehicle_context={"model": "BMW"}
        )
        data = request.model_dump()
        assert data["fault_codes"] == ["P0301"]
        assert "vehicle_context" in data
        assert data["vehicle_context"]["model"] == "BMW"
    
    def test_query_request_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "fault_codes": ["P0301", "P0302"],
            "obd_data": {"rpm": 2000},
            "vehicle_context": {"model": "BMW"},
            "session_id": "session_123"
        }
        request = QueryRequest(**data)
        assert len(request.fault_codes) == 2
        assert request.session_id == "session_123"


class TestQueryResponse:
    """Test QueryResponse model."""
    
    def test_valid_query_response(self):
        """Test creating a valid query response."""
        recommendations = [
            Recommendation(
                id="rec_1",
                title="Guide 1",
                procedure_name="Proc 1",
                score=0.9
            )
        ]
        response = QueryResponse(
            recommendations=recommendations,
            needs_clarification=False,
            clarification_questions=None,
            session_id="session_123",
            query_text="Test query"
        )
        assert len(response.recommendations) == 1
        assert response.needs_clarification is False
        assert response.clarification_questions is None
    
    def test_query_response_with_clarification(self):
        """Test query response with clarification questions."""
        response = QueryResponse(
            recommendations=[],
            needs_clarification=True,
            clarification_questions=["Question 1", "Question 2"],
            session_id="session_456",
            query_text="Test"
        )
        assert response.needs_clarification is True
        assert len(response.clarification_questions) == 2
    
    def test_query_response_serialization(self):
        """Test serialization to dict."""
        response = QueryResponse(
            recommendations=[],
            needs_clarification=False,
            session_id="session_789",
            query_text="Test query"
        )
        data = response.model_dump()
        assert data["session_id"] == "session_789"
        assert data["needs_clarification"] is False
        assert isinstance(data["recommendations"], list)


class TestClarifyRequest:
    """Test ClarifyRequest model."""
    
    def test_valid_clarify_request(self):
        """Test creating a valid clarify request."""
        request = ClarifyRequest(
            session_id="session_123",
            responses=["Answer 1", "Answer 2"]
        )
        assert request.session_id == "session_123"
        assert len(request.responses) == 2
    
    def test_clarify_request_empty_session_id_validation(self):
        """Test that session_id cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            ClarifyRequest(
                session_id="",
                responses=["Answer 1"]
            )
        assert "at least 1 character" in str(exc_info.value)
    
    def test_clarify_request_empty_responses_validation(self):
        """Test that responses cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            ClarifyRequest(
                session_id="session_123",
                responses=[]
            )
        assert "at least 1 item" in str(exc_info.value)
    
    def test_clarify_request_serialization(self):
        """Test serialization to dict."""
        request = ClarifyRequest(
            session_id="session_123",
            responses=["Answer 1"]
        )
        data = request.model_dump()
        assert data["session_id"] == "session_123"
        assert data["responses"] == ["Answer 1"]


class TestRepairOutcome:
    """Test RepairOutcome enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert RepairOutcome.SUCCESS.value == "success"
        assert RepairOutcome.FAILURE.value == "failure"
        assert RepairOutcome.PARTIAL.value == "partial"


class TestRatingFeedback:
    """Test RatingFeedback model."""
    
    def test_valid_rating_feedback(self):
        """Test creating a valid rating feedback."""
        feedback = RatingFeedback(
            session_id="session_123",
            rating=5,
            selected_guide="guide_456"
        )
        assert feedback.rating == 5
        assert feedback.selected_guide == "guide_456"
    
    def test_rating_feedback_without_selected_guide(self):
        """Test rating feedback without selected guide."""
        feedback = RatingFeedback(
            session_id="session_123",
            rating=3
        )
        assert feedback.selected_guide is None
    
    def test_rating_validation_min(self):
        """Test that rating must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            RatingFeedback(
                session_id="session_123",
                rating=0
            )
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_rating_validation_max(self):
        """Test that rating must be <= 5."""
        with pytest.raises(ValidationError) as exc_info:
            RatingFeedback(
                session_id="session_123",
                rating=6
            )
        assert "less than or equal to 5" in str(exc_info.value)
    
    def test_rating_feedback_serialization(self):
        """Test serialization to dict."""
        feedback = RatingFeedback(
            session_id="session_123",
            rating=4
        )
        data = feedback.model_dump()
        assert data["rating"] == 4
        assert data["selected_guide"] is None


class TestRepairOutcomeFeedback:
    """Test RepairOutcomeFeedback model."""
    
    def test_valid_repair_outcome_feedback(self):
        """Test creating a valid repair outcome feedback."""
        feedback = RepairOutcomeFeedback(
            session_id="session_123",
            outcome=RepairOutcome.SUCCESS,
            details={"notes": "Worked perfectly"}
        )
        assert feedback.outcome == RepairOutcome.SUCCESS
        assert feedback.details["notes"] == "Worked perfectly"
    
    def test_repair_outcome_feedback_without_details(self):
        """Test repair outcome feedback without details."""
        feedback = RepairOutcomeFeedback(
            session_id="session_123",
            outcome=RepairOutcome.FAILURE
        )
        assert feedback.details is None
    
    def test_repair_outcome_enum_validation(self):
        """Test that outcome must be a valid enum value."""
        # Valid enum value
        feedback = RepairOutcomeFeedback(
            session_id="session_123",
            outcome="success"  # String value should work
        )
        assert feedback.outcome == RepairOutcome.SUCCESS
        
        # Invalid value
        with pytest.raises(ValidationError):
            RepairOutcomeFeedback(
                session_id="session_123",
                outcome="invalid_outcome"
            )
    
    def test_repair_outcome_feedback_serialization(self):
        """Test serialization to dict."""
        feedback = RepairOutcomeFeedback(
            session_id="session_123",
            outcome=RepairOutcome.PARTIAL
        )
        data = feedback.model_dump()
        assert data["outcome"] == "partial"
        assert isinstance(data, dict)


class TestConversationCorrection:
    """Test ConversationCorrection model."""
    
    def test_valid_conversation_correction(self):
        """Test creating a valid conversation correction."""
        correction = ConversationCorrection(
            session_id="session_123",
            correction={"field": "question", "value": "Corrected question"}
        )
        assert correction.correction["field"] == "question"
    
    def test_conversation_correction_empty_session_id(self):
        """Test that session_id cannot be empty."""
        with pytest.raises(ValidationError):
            ConversationCorrection(
                session_id="",
                correction={}
            )
    
    def test_conversation_correction_serialization(self):
        """Test serialization to dict."""
        correction = ConversationCorrection(
            session_id="session_123",
            correction={"key": "value"}
        )
        data = correction.model_dump()
        assert data["session_id"] == "session_123"
        assert data["correction"]["key"] == "value"


class TestFeedbackStatistics:
    """Test FeedbackStatistics model."""
    
    def test_valid_feedback_statistics(self):
        """Test creating valid feedback statistics."""
        stats = FeedbackStatistics(
            total_sessions=100,
            rated_sessions=80,
            average_rating=4.2,
            repair_outcomes={"success": 50, "failure": 20, "partial": 10},
            corrected_sessions=5,
            rating_coverage=0.8
        )
        assert stats.total_sessions == 100
        assert stats.average_rating == 4.2
        assert stats.repair_outcomes["success"] == 50
    
    def test_feedback_statistics_validation_negative(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            FeedbackStatistics(
                total_sessions=-1,
                rated_sessions=0,
                average_rating=0.0,
                repair_outcomes={},
                corrected_sessions=0,
                rating_coverage=0.0
            )
    
    def test_average_rating_validation(self):
        """Test that average_rating must be <= 5.0."""
        with pytest.raises(ValidationError):
            FeedbackStatistics(
                total_sessions=10,
                rated_sessions=10,
                average_rating=6.0,
                repair_outcomes={},
                corrected_sessions=0,
                rating_coverage=1.0
            )
    
    def test_rating_coverage_validation(self):
        """Test that rating_coverage must be <= 1.0."""
        with pytest.raises(ValidationError):
            FeedbackStatistics(
                total_sessions=10,
                rated_sessions=10,
                average_rating=4.0,
                repair_outcomes={},
                corrected_sessions=0,
                rating_coverage=1.5
            )
    
    def test_feedback_statistics_serialization(self):
        """Test serialization to dict."""
        stats = FeedbackStatistics(
            total_sessions=50,
            rated_sessions=40,
            average_rating=4.0,
            repair_outcomes={},
            corrected_sessions=2,
            rating_coverage=0.8
        )
        data = stats.model_dump()
        assert data["total_sessions"] == 50
        assert data["rating_coverage"] == 0.8


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_query_request_none_vehicle_context(self):
        """Test that vehicle_context can be None."""
        request = QueryRequest(
            fault_codes=["P0301"],
            obd_data={},
            vehicle_context=None
        )
        assert request.vehicle_context is None
    
    def test_query_response_empty_recommendations(self):
        """Test query response with empty recommendations list."""
        response = QueryResponse(
            recommendations=[],
            needs_clarification=True,
            session_id="session_123",
            query_text="Test"
        )
        assert len(response.recommendations) == 0
    
    def test_recommendation_score_boundaries(self):
        """Test recommendation score at boundaries."""
        # Minimum valid score
        rec_min = Recommendation(
            id="rec_1",
            title="Test",
            procedure_name="Test",
            score=0.0
        )
        assert rec_min.score == 0.0
        
        # Maximum valid score
        rec_max = Recommendation(
            id="rec_2",
            title="Test",
            procedure_name="Test",
            score=1.0
        )
        assert rec_max.score == 1.0
    
    def test_rating_boundaries(self):
        """Test rating at boundaries."""
        # Minimum valid rating
        feedback_min = RatingFeedback(
            session_id="session_123",
            rating=1
        )
        assert feedback_min.rating == 1
        
        # Maximum valid rating
        feedback_max = RatingFeedback(
            session_id="session_123",
            rating=5
        )
        assert feedback_max.rating == 5

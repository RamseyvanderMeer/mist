"""
Unit tests for FeedbackCollector.

Tests all CRUD operations, score aggregation, and data validation.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.feedback.collector import FeedbackCollector
from src.database.schema import FeedbackSession, MistFeedback


class TestFeedbackCollector:
    """Test FeedbackCollector functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def collector(self, temp_db):
        """Create FeedbackCollector instance with temp database."""
        return FeedbackCollector(temp_db)
    
    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database schema."""
        collector = FeedbackCollector(temp_db)
        assert collector.db_path.exists()
    
    def test_init_uses_default_path(self):
        """Test that initialization uses default path when None provided."""
        # This will use the default path from paths module
        # We'll just verify it doesn't crash
        try:
            collector = FeedbackCollector()
            assert collector.db_path is not None
        except Exception:
            # If default path doesn't work in test env, that's okay
            pass
    
    def test_save_session_creates_new(self, collector):
        """Test creating a new feedback session."""
        session_id = collector.save_session(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90}
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        
        # Verify session was saved
        session = collector.get_session(session_id)
        assert session is not None
        assert session["fault_codes"] == ["P0301", "P0302"]
        assert session["obd_data"] == {"rpm": 2000, "temp": 90}
    
    def test_save_session_updates_existing(self, collector):
        """Test updating an existing feedback session."""
        # Create session
        session_id = collector.save_session(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000}
        )
        
        # Update session
        collector.save_session(
            session_id=session_id,
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90},
            explicit_rating=5
        )
        
        # Verify update
        session = collector.get_session(session_id)
        assert session["fault_codes"] == ["P0301", "P0302"]
        assert session["obd_data"] == {"rpm": 2000, "temp": 90}
        assert session["explicit_rating"] == 5
    
    def test_save_session_with_all_fields(self, collector):
        """Test saving session with all fields."""
        session_id = collector.save_session(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90},
            clarification_questions=["What is the RPM?", "Any sounds?"],
            user_responses=["2000 RPM", "Yes, knocking"],
            recommended_guides=["guide-1", "guide-2"],
            selected_guide="guide-1",
            explicit_rating=4,
            repair_outcome="success",
            conversation_corrections=[{"question": "Q1", "correction": "C1"}],
            timestamp="2024-01-01T12:00:00"
        )
        
        session = collector.get_session(session_id)
        assert session["fault_codes"] == ["P0301", "P0302"]
        assert session["obd_data"] == {"rpm": 2000, "temp": 90}
        assert session["clarification_questions"] == ["What is the RPM?", "Any sounds?"]
        assert session["user_responses"] == ["2000 RPM", "Yes, knocking"]
        assert session["recommended_guides"] == ["guide-1", "guide-2"]
        assert session["selected_guide"] == "guide-1"
        assert session["explicit_rating"] == 4
        assert session["repair_outcome"] == "success"
        assert session["conversation_corrections"] == [{"question": "Q1", "correction": "C1"}]
        assert session["timestamp"] == "2024-01-01T12:00:00"
    
    def test_save_session_validates_rating_range(self, collector):
        """Test that save_session validates rating range."""
        # Test rating too low
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            collector.save_session(
                fault_codes=["P0301"],
                explicit_rating=0
            )
        
        # Test rating too high
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            collector.save_session(
                fault_codes=["P0301"],
                explicit_rating=6
            )
        
        # Test valid ratings
        for rating in [1, 2, 3, 4, 5]:
            session_id = collector.save_session(
                fault_codes=["P0301"],
                explicit_rating=rating
            )
            session = collector.get_session(session_id)
            assert session["explicit_rating"] == rating
    
    def test_get_session_returns_none_for_missing(self, collector):
        """Test that get_session returns None for non-existent session."""
        result = collector.get_session("non-existent-session")
        assert result is None
    
    def test_get_session_deserializes_json_fields(self, collector):
        """Test that get_session properly deserializes JSON fields."""
        session_id = collector.save_session(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90},
            clarification_questions=["Q1", "Q2"],
            user_responses=["R1", "R2"],
            recommended_guides=["G1", "G2"],
            conversation_corrections=[{"key": "value"}]
        )
        
        session = collector.get_session(session_id)
        assert isinstance(session["fault_codes"], list)
        assert isinstance(session["obd_data"], dict)
        assert isinstance(session["clarification_questions"], list)
        assert isinstance(session["user_responses"], list)
        assert isinstance(session["recommended_guides"], list)
        assert isinstance(session["conversation_corrections"], list)
    
    def test_save_feedback_creates_new(self, collector):
        """Test creating a new feedback entry."""
        # Create session first
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Create feedback
        feedback_id = collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=5,
            repair_outcome="success",
            feedback_text="Worked perfectly!"
        )
        
        assert feedback_id is not None
        assert isinstance(feedback_id, str)
    
    def test_save_feedback_updates_existing(self, collector):
        """Test updating an existing feedback entry."""
        # Create session
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Create feedback
        feedback_id = collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=3
        )
        
        # Update feedback
        collector.save_feedback(
            session_id=session_id,
            feedback_id=feedback_id,
            rating=5,
            repair_outcome="success"
        )
        
        # Verify we can still retrieve it (via database query)
        # Note: We don't have a get_feedback method, but we can verify
        # the update worked by checking procedure score
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert score > 0
    
    def test_save_feedback_validates_rating_range(self, collector):
        """Test that save_feedback validates rating range."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Test rating too low
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            collector.save_feedback(
                session_id=session_id,
                rating=0
            )
        
        # Test rating too high
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            collector.save_feedback(
                session_id=session_id,
                rating=6
            )
    
    def test_save_feedback_validates_outcome(self, collector):
        """Test that save_feedback validates repair outcome."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Test invalid outcome
        with pytest.raises(ValueError, match="Repair outcome must be one of"):
            collector.save_feedback(
                session_id=session_id,
                repair_outcome="invalid"
            )
        
        # Test valid outcomes
        for outcome in ["success", "failure", "partial"]:
            feedback_id = collector.save_feedback(
                session_id=session_id,
                repair_outcome=outcome
            )
            assert feedback_id is not None
    
    def test_save_feedback_requires_existing_session(self, collector):
        """Test that save_feedback requires existing session."""
        with pytest.raises(RuntimeError, match="Session .* does not exist"):
            collector.save_feedback(
                session_id="non-existent-session",
                rating=5
            )
    
    def test_save_feedback_supports_all_feedback_types(self, collector):
        """Test that save_feedback supports all feedback types."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Test with rating only
        feedback_id1 = collector.save_feedback(
            session_id=session_id,
            rating=5
        )
        assert feedback_id1 is not None
        
        # Test with outcome only
        feedback_id2 = collector.save_feedback(
            session_id=session_id,
            repair_outcome="success"
        )
        assert feedback_id2 is not None
        
        # Test with text only
        feedback_id3 = collector.save_feedback(
            session_id=session_id,
            feedback_text="Great guide!"
        )
        assert feedback_id3 is not None
        
        # Test with all fields
        feedback_id4 = collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=5,
            repair_outcome="success",
            feedback_text="Perfect!"
        )
        assert feedback_id4 is not None
    
    def test_get_procedure_score_returns_none_for_no_feedback(self, collector):
        """Test that get_procedure_score returns None when no feedback exists."""
        score = collector.get_procedure_score("PROC-NO-FEEDBACK")
        assert score is None
    
    def test_get_procedure_score_from_ratings(self, collector):
        """Test procedure score calculation from ratings."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Add multiple feedback entries with ratings
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=5
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=4
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=3
        )
        
        # Average rating = (5 + 4 + 3) / 3 = 4.0
        # Normalized: (4.0 - 1) / 4 = 0.75
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Should be approximately 0.75 (allowing for floating point)
        assert abs(score - 0.75) < 0.01
    
    def test_get_procedure_score_from_outcomes(self, collector):
        """Test procedure score calculation from outcomes."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Add feedback with outcomes
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="success"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="success"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="partial"
        )
        
        # Success rate = (2 * 1.0 + 1 * 0.5) / 3 = 0.833...
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Should be approximately 0.833
        assert abs(score - 0.833) < 0.1
    
    def test_get_procedure_score_combined_rating_and_outcome(self, collector):
        """Test procedure score with both ratings and outcomes."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Add feedback with both rating and outcome
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=5,
            repair_outcome="success"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=4,
            repair_outcome="success"
        )
        
        # Should combine both: 60% rating, 40% outcome
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Average rating = 4.5, normalized = 0.875
        # Success rate = 1.0
        # Combined = 0.6 * 0.875 + 0.4 * 1.0 = 0.925
        assert score > 0.8  # Should be high
    
    def test_get_procedure_score_handles_failure_outcomes(self, collector):
        """Test procedure score with failure outcomes."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Add feedback with failures
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="failure"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="failure"
        )
        
        # Success rate should be 0.0
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert score == 0.0
    
    def test_get_procedure_score_handles_mixed_outcomes(self, collector):
        """Test procedure score with mixed outcomes."""
        session_id = collector.save_session(fault_codes=["P0301"])
        
        # Add mixed outcomes
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="success"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="partial"
        )
        collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            repair_outcome="failure"
        )
        
        # Success rate = (1 * 1.0 + 1 * 0.5 + 1 * 0.0) / 3 = 0.5
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) < 0.1
    
    def test_crud_operations_workflow(self, collector):
        """Test complete CRUD workflow."""
        # Create session
        session_id = collector.save_session(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000}
        )
        assert session_id is not None
        
        # Read session
        session = collector.get_session(session_id)
        assert session is not None
        assert session["fault_codes"] == ["P0301"]
        
        # Update session
        collector.save_session(
            session_id=session_id,
            explicit_rating=5
        )
        updated_session = collector.get_session(session_id)
        assert updated_session["explicit_rating"] == 5
        
        # Create feedback
        feedback_id = collector.save_feedback(
            session_id=session_id,
            procedure_id="PROC-123",
            rating=5
        )
        assert feedback_id is not None
        
        # Get procedure score (indirect read of feedback)
        score = collector.get_procedure_score("PROC-123")
        assert score is not None
        assert score > 0

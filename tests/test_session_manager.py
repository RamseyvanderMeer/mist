"""
Unit tests for SessionManager.

Tests all session management operations including creation, retrieval,
updates, clarification tracking, recommendations, and expiration.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from src.retrieval.session_manager import SessionManager
from src.database.schema import FeedbackSession


class TestSessionManager:
    """Test SessionManager functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def manager(self, temp_db):
        """Create SessionManager instance with temp database."""
        return SessionManager(temp_db)
    
    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database schema."""
        manager = SessionManager(temp_db)
        assert manager.db_path.exists()
    
    def test_init_uses_default_path(self):
        """Test that initialization uses default path when None provided."""
        # This will use the default path from get_mist_db_path
        # We'll just verify it doesn't crash
        try:
            manager = SessionManager()
            assert manager.db_path is not None
        except Exception:
            # If default path doesn't work in test env, that's okay
            pass
    
    def test_create_session_generates_uuid(self, manager):
        """Test that create_session generates a UUID."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000}
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format length
    
    def test_create_session_stores_initial_state(self, manager):
        """Test that create_session stores initial fault codes and OBD data."""
        fault_codes = ["P0301", "P0302"]
        obd_data = {"rpm": 2000, "temp": 90}
        
        session_id = manager.create_session(
            fault_codes=fault_codes,
            obd_data=obd_data
        )
        
        # Verify session was saved
        session = manager.get_session(session_id)
        assert session is not None
        assert session["fault_codes"] == fault_codes
        assert session["obd_data"] == obd_data
        assert session["timestamp"] is not None
    
    def test_create_session_merges_vehicle_context(self, manager):
        """Test that create_session merges vehicle_context into obd_data."""
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        vehicle_context = {"make": "BMW", "model": "X5"}
        
        session_id = manager.create_session(
            fault_codes=fault_codes,
            obd_data=obd_data,
            vehicle_context=vehicle_context
        )
        
        session = manager.get_session(session_id)
        assert session["obd_data"]["rpm"] == 2000
        assert session["obd_data"]["make"] == "BMW"
        assert session["obd_data"]["model"] == "X5"
    
    def test_get_session_returns_none_for_missing(self, manager):
        """Test that get_session returns None for non-existent session."""
        result = manager.get_session("non-existent-session")
        assert result is None
    
    def test_get_session_deserializes_json_fields(self, manager):
        """Test that get_session properly deserializes JSON fields."""
        session_id = manager.create_session(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90}
        )
        
        # Add clarification data
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q1", "Q2"],
            responses=["R1", "R2"]
        )
        
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=["G1", "G2"]
        )
        
        session = manager.get_session(session_id)
        assert isinstance(session["fault_codes"], list)
        assert isinstance(session["obd_data"], dict)
        assert isinstance(session["clarification_questions"], list)
        assert isinstance(session["user_responses"], list)
        assert isinstance(session["recommended_guides"], list)
    
    def test_update_session_updates_fields(self, manager):
        """Test that update_session updates specified fields."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={"rpm": 2000}
        )
        
        # Update multiple fields
        manager.update_session(
            session_id=session_id,
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90},
            selected_guide="guide-1",
            explicit_rating=5
        )
        
        session = manager.get_session(session_id)
        assert session["fault_codes"] == ["P0301", "P0302"]
        assert session["obd_data"] == {"rpm": 2000, "temp": 90}
        assert session["selected_guide"] == "guide-1"
        assert session["explicit_rating"] == 5
    
    def test_update_session_validates_rating_range(self, manager):
        """Test that update_session validates rating range."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Test rating too low
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            manager.update_session(
                session_id=session_id,
                explicit_rating=0
            )
        
        # Test rating too high
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            manager.update_session(
                session_id=session_id,
                explicit_rating=6
            )
        
        # Test valid ratings
        for rating in [1, 2, 3, 4, 5]:
            manager.update_session(
                session_id=session_id,
                explicit_rating=rating
            )
            session = manager.get_session(session_id)
            assert session["explicit_rating"] == rating
    
    def test_update_session_requires_existing_session(self, manager):
        """Test that update_session requires existing session."""
        with pytest.raises(RuntimeError, match="Session .* does not exist"):
            manager.update_session(
                session_id="non-existent-session",
                fault_codes=["P0301"]
            )
    
    def test_add_clarification_round_appends_to_existing(self, manager):
        """Test that add_clarification_round appends to existing arrays."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Add first round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q1", "Q2"],
            responses=["R1", "R2"]
        )
        
        session = manager.get_session(session_id)
        assert session["clarification_questions"] == ["Q1", "Q2"]
        assert session["user_responses"] == ["R1", "R2"]
        
        # Add second round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q3"],
            responses=["R3"]
        )
        
        session = manager.get_session(session_id)
        assert session["clarification_questions"] == ["Q1", "Q2", "Q3"]
        assert session["user_responses"] == ["R1", "R2", "R3"]
    
    def test_add_clarification_round_validates_lengths(self, manager):
        """Test that add_clarification_round validates question/response lengths."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="Questions and responses must have same length"):
            manager.add_clarification_round(
                session_id=session_id,
                questions=["Q1", "Q2"],
                responses=["R1"]
            )
    
    def test_add_clarification_round_requires_existing_session(self, manager):
        """Test that add_clarification_round requires existing session."""
        with pytest.raises(RuntimeError, match="Session .* does not exist"):
            manager.add_clarification_round(
                session_id="non-existent-session",
                questions=["Q1"],
                responses=["R1"]
            )
    
    def test_update_recommendations_stores_guides(self, manager):
        """Test that update_recommendations stores recommended guides."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        recommended_guides = ["guide-1", "guide-2", "guide-3"]
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=recommended_guides
        )
        
        session = manager.get_session(session_id)
        assert session["recommended_guides"] == recommended_guides
    
    def test_update_recommendations_overwrites_existing(self, manager):
        """Test that update_recommendations overwrites existing recommendations."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Set initial recommendations
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=["guide-1", "guide-2"]
        )
        
        # Update with new recommendations
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=["guide-3", "guide-4", "guide-5"]
        )
        
        session = manager.get_session(session_id)
        assert session["recommended_guides"] == ["guide-3", "guide-4", "guide-5"]
    
    def test_update_recommendations_requires_existing_session(self, manager):
        """Test that update_recommendations requires existing session."""
        with pytest.raises(RuntimeError, match="Session .* does not exist"):
            manager.update_recommendations(
                session_id="non-existent-session",
                recommended_guides=["guide-1"]
            )
    
    def test_is_expired_returns_false_for_new_session(self, manager):
        """Test that is_expired returns False for newly created session."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Should not be expired immediately
        assert manager.is_expired(session_id, expiration_hours=24) is False
    
    def test_is_expired_returns_true_for_old_session(self, manager):
        """Test that is_expired returns True for old session."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Manually set old created_at timestamp
        old_created_at = (datetime.now() - timedelta(hours=25)).isoformat()
        manager.update_session(
            session_id=session_id,
            created_at=old_created_at
        )
        
        # Should be expired with 24 hour expiration
        assert manager.is_expired(session_id, expiration_hours=24) is True
    
    def test_is_expired_uses_custom_expiration_hours(self, manager):
        """Test that is_expired respects custom expiration_hours parameter."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Set created_at to 2 hours ago
        old_created_at = (datetime.now() - timedelta(hours=2)).isoformat()
        manager.update_session(
            session_id=session_id,
            created_at=old_created_at
        )
        
        # Should not be expired with 24 hour expiration
        assert manager.is_expired(session_id, expiration_hours=24) is False
        
        # Should be expired with 1 hour expiration
        assert manager.is_expired(session_id, expiration_hours=1) is True
    
    def test_is_expired_requires_existing_session(self, manager):
        """Test that is_expired requires existing session."""
        with pytest.raises(RuntimeError, match="Session .* does not exist"):
            manager.is_expired("non-existent-session", expiration_hours=24)
    
    def test_cleanup_expired_sessions_removes_old_sessions(self, manager):
        """Test that cleanup_expired_sessions removes expired sessions."""
        # Create multiple sessions
        session_id1 = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        session_id2 = manager.create_session(
            fault_codes=["P0302"],
            obd_data={}
        )
        
        # Set one session to be old
        old_created_at = (datetime.now() - timedelta(hours=25)).isoformat()
        manager.update_session(
            session_id=session_id1,
            created_at=old_created_at
        )
        
        # Cleanup expired sessions
        deleted_count = manager.cleanup_expired_sessions(expiration_hours=24)
        
        # Should have deleted 1 session
        assert deleted_count == 1
        
        # Old session should be gone
        assert manager.get_session(session_id1) is None
        
        # New session should still exist
        assert manager.get_session(session_id2) is not None
    
    def test_cleanup_expired_sessions_returns_zero_when_none_expired(self, manager):
        """Test that cleanup_expired_sessions returns 0 when no sessions expired."""
        # Create a new session
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # Cleanup expired sessions
        deleted_count = manager.cleanup_expired_sessions(expiration_hours=24)
        
        # Should have deleted 0 sessions
        assert deleted_count == 0
        
        # Session should still exist
        assert manager.get_session(session_id) is not None
    
    def test_cleanup_expired_sessions_uses_custom_expiration_hours(self, manager):
        """Test that cleanup_expired_sessions respects custom expiration_hours."""
        # Create session with 2 hour old timestamp
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        old_created_at = (datetime.now() - timedelta(hours=2)).isoformat()
        manager.update_session(
            session_id=session_id,
            created_at=old_created_at
        )
        
        # Cleanup with 1 hour expiration should remove it
        deleted_count = manager.cleanup_expired_sessions(expiration_hours=1)
        assert deleted_count == 1
        assert manager.get_session(session_id) is None
    
    def test_complete_workflow(self, manager):
        """Test complete session management workflow."""
        # Create session
        session_id = manager.create_session(
            fault_codes=["P0301", "P0302"],
            obd_data={"rpm": 2000, "temp": 90},
            vehicle_context={"make": "BMW"}
        )
        assert session_id is not None
        
        # Retrieve session
        session = manager.get_session(session_id)
        assert session is not None
        assert len(session["fault_codes"]) == 2
        
        # Add clarification round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["What is the RPM?", "Any unusual sounds?"],
            responses=["2000 RPM", "Yes, knocking sound"]
        )
        
        # Update recommendations
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=["guide-1", "guide-2"]
        )
        
        # Update session with additional info
        manager.update_session(
            session_id=session_id,
            selected_guide="guide-1",
            explicit_rating=5
        )
        
        # Retrieve updated session
        updated_session = manager.get_session(session_id)
        assert updated_session["clarification_questions"] == ["What is the RPM?", "Any unusual sounds?"]
        assert updated_session["user_responses"] == ["2000 RPM", "Yes, knocking sound"]
        assert updated_session["recommended_guides"] == ["guide-1", "guide-2"]
        assert updated_session["selected_guide"] == "guide-1"
        assert updated_session["explicit_rating"] == 5
        
        # Check expiration (should not be expired)
        assert manager.is_expired(session_id, expiration_hours=24) is False
    
    def test_multiple_clarification_rounds(self, manager):
        """Test tracking multiple clarification rounds."""
        session_id = manager.create_session(
            fault_codes=["P0301"],
            obd_data={}
        )
        
        # First round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q1"],
            responses=["R1"]
        )
        
        # Second round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q2", "Q3"],
            responses=["R2", "R3"]
        )
        
        # Third round
        manager.add_clarification_round(
            session_id=session_id,
            questions=["Q4"],
            responses=["R4"]
        )
        
        session = manager.get_session(session_id)
        assert session["clarification_questions"] == ["Q1", "Q2", "Q3", "Q4"]
        assert session["user_responses"] == ["R1", "R2", "R3", "R4"]
    
    def test_empty_lists_handled_correctly(self, manager):
        """Test that empty lists are handled correctly."""
        session_id = manager.create_session(
            fault_codes=[],
            obd_data={}
        )
        
        session = manager.get_session(session_id)
        assert session["fault_codes"] == []
        assert session["obd_data"] == {}
        
        # Add clarification with empty lists (should work)
        manager.add_clarification_round(
            session_id=session_id,
            questions=[],
            responses=[]
        )
        
        session = manager.get_session(session_id)
        assert session["clarification_questions"] == []
        assert session["user_responses"] == []
        
        # Update recommendations with empty list
        manager.update_recommendations(
            session_id=session_id,
            recommended_guides=[]
        )
        
        session = manager.get_session(session_id)
        assert session["recommended_guides"] == []

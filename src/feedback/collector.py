"""
Feedback collection and storage system.

Uses SQLAlchemy ORM models for database operations.
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from ..database.connection import create_connection
from ..database.migrations import init_database
from ..database.schema import FeedbackSession, MistFeedback
from ..paths import get_paths

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and stores user feedback for self-improvement.
    
    Uses SQLAlchemy ORM models for database operations with proper
    session management and error handling.
    """
    
    def __init__(self, db_path: str | None = None):
        """
        Initialize feedback collector.
        
        Args:
            db_path: Path to feedback database. If None, uses default from paths module.
        
        Raises:
            RuntimeError: If database initialization fails
        """
        if db_path is None:
            paths = get_paths()
            db_path = str(paths.feedback_db)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        if not init_database(str(self.db_path)):
            raise RuntimeError(f"Failed to initialize database at {self.db_path}")
        
        # Create database connection
        self._connection = create_connection(self.db_path)
        
        logger.info(f"Initialized FeedbackCollector with database: {self.db_path}")
    
    def save_session(
        self,
        session_id: str | None = None,
        fault_codes: list[str] | None = None,
        obd_data: dict[str, Any] | None = None,
        clarification_questions: list[str] | None = None,
        user_responses: list[str] | None = None,
        recommended_guides: list[str] | None = None,
        selected_guide: str | None = None,
        explicit_rating: int | None = None,
        repair_outcome: str | None = None,
        conversation_corrections: list[dict[str, Any]] | None = None,
        timestamp: str | None = None,
    ) -> str:
        """
        Create or update a feedback session.
        
        Args:
            session_id: Unique session identifier. If None, generates a new UUID.
            fault_codes: List of fault codes (e.g., ["P0301", "P0302"])
            obd_data: Dictionary of OBD sensor data
            clarification_questions: List of clarification questions asked
            user_responses: List of user responses to clarification questions
            recommended_guides: List of recommended repair guide IDs
            selected_guide: ID of the guide that was actually selected
            explicit_rating: Rating from 1-5
            repair_outcome: Outcome string (success/failure/partial)
            conversation_corrections: List of correction dictionaries
            timestamp: ISO format timestamp. If None, uses current time.
        
        Returns:
            session_id: The session identifier (new or existing)
        
        Raises:
            ValueError: If rating is not in valid range (1-5)
            SQLAlchemyError: If database operation fails
        """
        # Validate rating if provided
        if explicit_rating is not None and not (1 <= explicit_rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5, got {explicit_rating}")
        
        # Generate session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        try:
            with self._connection.session() as session:
                # Check if session exists
                existing_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if existing_session:
                    # Update existing session
                    feedback_session = existing_session
                else:
                    # Create new session
                    feedback_session = FeedbackSession(session_id=session_id)
                    session.add(feedback_session)
                
                # Update fields using helper methods
                if fault_codes is not None:
                    feedback_session.set_fault_codes(fault_codes)
                if obd_data is not None:
                    feedback_session.set_obd_data(obd_data)
                if clarification_questions is not None:
                    feedback_session.set_clarification_questions(clarification_questions)
                if user_responses is not None:
                    feedback_session.set_user_responses(user_responses)
                if recommended_guides is not None:
                    feedback_session.set_recommended_guides(recommended_guides)
                if selected_guide is not None:
                    feedback_session.selected_guide = selected_guide
                if explicit_rating is not None:
                    feedback_session.explicit_rating = explicit_rating
                if repair_outcome is not None:
                    feedback_session.repair_outcome = repair_outcome
                if conversation_corrections is not None:
                    feedback_session.set_conversation_corrections(conversation_corrections)
                if timestamp is not None:
                    feedback_session.timestamp = timestamp
                
                session.commit()
                logger.debug(f"Saved session {session_id}")
                return session_id
                
        except SQLAlchemyError as e:
            logger.error(f"Database error saving session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving session {session_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Retrieve a feedback session by ID.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Dictionary with session data (deserialized JSON fields) or None if not found
        
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    logger.debug(f"Session {session_id} not found")
                    return None
                
                # Build result dictionary using helper methods
                result = {
                    "session_id": feedback_session.session_id,
                    "fault_codes": feedback_session.get_fault_codes(),
                    "obd_data": feedback_session.get_obd_data(),
                    "clarification_questions": feedback_session.get_clarification_questions(),
                    "user_responses": feedback_session.get_user_responses(),
                    "recommended_guides": feedback_session.get_recommended_guides(),
                    "selected_guide": feedback_session.selected_guide,
                    "explicit_rating": feedback_session.explicit_rating,
                    "repair_outcome": feedback_session.repair_outcome,
                    "conversation_corrections": feedback_session.get_conversation_corrections(),
                    "timestamp": feedback_session.timestamp,
                    "created_at": feedback_session.created_at,
                }
                
                return result
                
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving session {session_id}: {e}")
            raise
    
    def save_feedback(
        self,
        session_id: str,
        procedure_id: str | None = None,
        rating: int | None = None,
        repair_outcome: str | None = None,
        feedback_text: str | None = None,
        feedback_id: str | None = None,
    ) -> str:
        """
        Save individual feedback entry.
        
        Supports all feedback types: ratings (1-5), outcomes (success/failure/partial),
        and text feedback.
        
        Args:
            session_id: Session identifier (must exist)
            procedure_id: Procedure identifier (optional)
            rating: Rating from 1-5 (optional)
            repair_outcome: Outcome string - success/failure/partial (optional)
            feedback_text: Free-form feedback text (optional)
            feedback_id: Unique feedback identifier. If None, generates a new UUID.
        
        Returns:
            feedback_id: The feedback identifier (new or existing)
        
        Raises:
            ValueError: If rating is not in valid range (1-5) or outcome is invalid
            RuntimeError: If session_id does not exist
            SQLAlchemyError: If database operation fails
        """
        # Validate rating if provided
        if rating is not None and not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")
        
        # Validate repair_outcome if provided
        valid_outcomes = {"success", "failure", "partial"}
        if repair_outcome is not None and repair_outcome not in valid_outcomes:
            raise ValueError(
                f"Repair outcome must be one of {valid_outcomes}, got {repair_outcome}"
            )
        
        # Generate feedback_id if not provided
        if feedback_id is None:
            feedback_id = str(uuid.uuid4())
        
        try:
            with self._connection.session() as session:
                # Verify session exists
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    raise RuntimeError(f"Session {session_id} does not exist")
                
                # Check if feedback entry already exists
                existing_feedback = session.query(MistFeedback).filter_by(
                    feedback_id=feedback_id
                ).first()
                
                if existing_feedback:
                    # Update existing feedback
                    mist_feedback = existing_feedback
                else:
                    # Create new feedback entry
                    mist_feedback = MistFeedback(
                        feedback_id=feedback_id,
                        session_id=session_id
                    )
                    session.add(mist_feedback)
                
                # Update fields
                if procedure_id is not None:
                    mist_feedback.procedure_id = procedure_id
                if rating is not None:
                    mist_feedback.rating = rating
                if repair_outcome is not None:
                    mist_feedback.repair_outcome = repair_outcome
                if feedback_text is not None:
                    mist_feedback.feedback_text = feedback_text
                
                session.commit()
                logger.debug(f"Saved feedback {feedback_id} for session {session_id}")
                return feedback_id
                
        except SQLAlchemyError as e:
            logger.error(f"Database error saving feedback for session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving feedback for session {session_id}: {e}")
            raise
    
    def get_procedure_score(self, procedure_id: str) -> float | None:
        """
        Get aggregated feedback score for a procedure.
        
        Calculates a combined score from:
        - Average rating (normalized to 0.0-1.0)
        - Success rate from repair_outcome
        
        Args:
            procedure_id: Procedure identifier
        
        Returns:
            Aggregated score (0.0-1.0) or None if no feedback exists
        
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Get all feedback entries for this procedure
                feedback_entries = session.query(MistFeedback).filter_by(
                    procedure_id=procedure_id
                ).all()
                
                if not feedback_entries:
                    logger.debug(f"No feedback found for procedure {procedure_id}")
                    return None
                
                # Calculate average rating
                ratings = [f.rating for f in feedback_entries if f.rating is not None]
                avg_rating = None
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    # Normalize from 1-5 scale to 0.0-1.0
                    avg_rating = (avg_rating - 1) / 4.0
                
                # Calculate success rate
                outcomes = [f.repair_outcome for f in feedback_entries if f.repair_outcome]
                success_rate: float | None = None
                if outcomes:
                    success_count = sum(1 for o in outcomes if o == "success")
                    partial_count = sum(1 for o in outcomes if o == "partial")
                    # Weight: success=1.0, partial=0.5, failure=0.0
                    success_rate = (success_count + 0.5 * partial_count) / len(outcomes)
                
                # Combine scores (weighted average)
                # If both available: 60% rating, 40% outcome
                # If only one available: use that
                if avg_rating is not None and success_rate is not None:
                    combined_score = 0.6 * avg_rating + 0.4 * success_rate
                elif avg_rating is not None:
                    combined_score = avg_rating
                elif success_rate is not None:
                    combined_score = success_rate
                else:
                    # No usable feedback data
                    return None
                
                # Ensure score is in valid range
                combined_score = max(0.0, min(1.0, combined_score))
                
                logger.debug(
                    f"Procedure {procedure_id} score: {combined_score:.3f} "
                    f"(ratings: {len(ratings)}, outcomes: {len(outcomes)})"
                )
                return combined_score
                
        except SQLAlchemyError as e:
            logger.error(f"Database error calculating score for procedure {procedure_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calculating score for procedure {procedure_id}: {e}")
            raise

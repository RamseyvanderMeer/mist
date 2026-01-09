"""
Session management for multi-turn conversational RAG.

Manages session state including fault codes, OBD data, clarification questions,
and user responses for tracking multi-turn conversations.
"""
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError

from ..database import get_mist_db_path, init_database
from ..database.connection import create_connection
from ..database.schema import FeedbackSession

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages multi-turn conversation sessions for conversational RAG.
    
    Tracks session state including fault codes, OBD data, clarification
    questions, and user responses. Uses the FeedbackSession model stored
    in the MIST database.
    """
    
    def __init__(self, db_path: str | None = None):
        """
        Initialize session manager.
        
        Args:
            db_path: Path to MIST database. If None, uses default from paths module.
        
        Raises:
            RuntimeError: If database initialization fails
        """
        if db_path is None:
            db_path = str(get_mist_db_path())
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        if not init_database(str(self.db_path)):
            raise RuntimeError(f"Failed to initialize database at {self.db_path}")
        
        # Create database connection
        self._connection = create_connection(self.db_path)
        
        logger.info(f"Initialized SessionManager with database: {self.db_path}")
    
    def create_session(
        self,
        fault_codes: List[str],
        obd_data: Dict[str, Any],
        vehicle_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session with UUID.
        
        Args:
            fault_codes: List of fault code strings (e.g., ["P0301", "P0302"])
            obd_data: Dictionary of OBD sensor data
            vehicle_context: Optional vehicle information (stored in obd_data if provided)
        
        Returns:
            session_id: The generated session UUID
        
        Raises:
            SQLAlchemyError: If database operation fails
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Merge vehicle_context into obd_data if provided
        if vehicle_context:
            obd_data = {**obd_data, **vehicle_context}
        
        try:
            with self._connection.session() as session:
                # Create new session
                feedback_session = FeedbackSession(session_id=session_id)
                session.add(feedback_session)
                
                # Set initial state
                feedback_session.set_fault_codes(fault_codes)
                feedback_session.set_obd_data(obd_data)
                feedback_session.timestamp = timestamp
                
                session.commit()
                logger.debug(f"Created new session {session_id}")
                return session_id
                
        except SQLAlchemyError as e:
            logger.error(f"Database error creating session: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.
        
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
    
    def update_session(self, session_id: str, **kwargs) -> None:
        """
        Update session fields.
        
        Args:
            session_id: Unique session identifier
            **kwargs: Fields to update. Supported fields:
                - fault_codes: List[str]
                - obd_data: Dict[str, Any]
                - clarification_questions: List[str]
                - user_responses: List[str]
                - recommended_guides: List[str]
                - selected_guide: str
                - explicit_rating: int (1-5)
                - repair_outcome: str
                - conversation_corrections: List[Dict[str, Any]]
                - timestamp: str
        
        Raises:
            RuntimeError: If session_id does not exist
            ValueError: If rating is not in valid range (1-5)
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Verify session exists
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    raise RuntimeError(f"Session {session_id} does not exist")
                
                # Update fields using helper methods
                if "fault_codes" in kwargs:
                    feedback_session.set_fault_codes(kwargs["fault_codes"])
                if "obd_data" in kwargs:
                    feedback_session.set_obd_data(kwargs["obd_data"])
                if "clarification_questions" in kwargs:
                    feedback_session.set_clarification_questions(kwargs["clarification_questions"])
                if "user_responses" in kwargs:
                    feedback_session.set_user_responses(kwargs["user_responses"])
                if "recommended_guides" in kwargs:
                    feedback_session.set_recommended_guides(kwargs["recommended_guides"])
                if "selected_guide" in kwargs:
                    feedback_session.selected_guide = kwargs["selected_guide"]
                if "explicit_rating" in kwargs:
                    rating = kwargs["explicit_rating"]
                    if not (1 <= rating <= 5):
                        raise ValueError(f"Rating must be between 1 and 5, got {rating}")
                    feedback_session.explicit_rating = rating
                if "repair_outcome" in kwargs:
                    feedback_session.repair_outcome = kwargs["repair_outcome"]
                if "conversation_corrections" in kwargs:
                    feedback_session.set_conversation_corrections(kwargs["conversation_corrections"])
                if "timestamp" in kwargs:
                    feedback_session.timestamp = kwargs["timestamp"]
                if "created_at" in kwargs:
                    feedback_session.created_at = kwargs["created_at"]
                
                session.commit()
                logger.debug(f"Updated session {session_id}")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error updating session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating session {session_id}: {e}")
            raise
    
    def add_clarification_round(
        self,
        session_id: str,
        questions: List[str],
        responses: List[str]
    ) -> None:
        """
        Add a clarification round to track clarification history.
        
        Appends new questions and responses to existing arrays to track
        multi-turn conversations.
        
        Args:
            session_id: Unique session identifier
            questions: List of clarification questions asked
            responses: List of user responses to questions
        
        Raises:
            RuntimeError: If session_id does not exist
            ValueError: If questions and responses lists have different lengths
            SQLAlchemyError: If database operation fails
        """
        if len(questions) != len(responses):
            raise ValueError(
                f"Questions and responses must have same length. "
                f"Got {len(questions)} questions and {len(responses)} responses"
            )
        
        try:
            with self._connection.session() as session:
                # Verify session exists
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    raise RuntimeError(f"Session {session_id} does not exist")
                
                # Get existing questions and responses
                existing_questions = feedback_session.get_clarification_questions()
                existing_responses = feedback_session.get_user_responses()
                
                # Append new questions and responses
                updated_questions = existing_questions + questions
                updated_responses = existing_responses + responses
                
                # Update session
                feedback_session.set_clarification_questions(updated_questions)
                feedback_session.set_user_responses(updated_responses)
                
                session.commit()
                logger.debug(
                    f"Added clarification round to session {session_id}: "
                    f"{len(questions)} questions/responses"
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Database error adding clarification round to session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding clarification round to session {session_id}: {e}")
            raise
    
    def update_recommendations(
        self,
        session_id: str,
        recommended_guides: List[str]
    ) -> None:
        """
        Update recommended repair guides for a session.
        
        Args:
            session_id: Unique session identifier
            recommended_guides: List of recommended repair guide IDs
        
        Raises:
            RuntimeError: If session_id does not exist
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Verify session exists
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    raise RuntimeError(f"Session {session_id} does not exist")
                
                # Update recommended guides
                feedback_session.set_recommended_guides(recommended_guides)
                
                session.commit()
                logger.debug(
                    f"Updated recommendations for session {session_id}: "
                    f"{len(recommended_guides)} guides"
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Database error updating recommendations for session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating recommendations for session {session_id}: {e}")
            raise
    
    def is_expired(self, session_id: str, expiration_hours: int = 24) -> bool:
        """
        Check if a session is expired based on creation time.
        
        Args:
            session_id: Unique session identifier
            expiration_hours: Number of hours before session expires (default: 24)
        
        Returns:
            True if session is expired, False otherwise
        
        Raises:
            RuntimeError: If session_id does not exist
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Verify session exists
                feedback_session = session.query(FeedbackSession).filter_by(
                    session_id=session_id
                ).first()
                
                if not feedback_session:
                    raise RuntimeError(f"Session {session_id} does not exist")
                
                # Parse created_at timestamp (prefer created_at, fall back to timestamp)
                if feedback_session.created_at:
                    created_time = datetime.fromisoformat(feedback_session.created_at)
                elif feedback_session.timestamp:
                    created_time = datetime.fromisoformat(feedback_session.timestamp)
                else:
                    return False  # Can't determine expiration
                
                # Check if expired
                expiration_time = created_time + timedelta(hours=expiration_hours)
                is_expired = datetime.now() > expiration_time
                
                logger.debug(
                    f"Session {session_id} expiration check: "
                    f"created={created_time}, expires={expiration_time}, expired={is_expired}"
                )
                
                return is_expired
                
        except SQLAlchemyError as e:
            logger.error(f"Database error checking expiration for session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking expiration for session {session_id}: {e}")
            raise
    
    def cleanup_expired_sessions(self, expiration_hours: int = 24) -> int:
        """
        Remove expired sessions from the database.
        
        Args:
            expiration_hours: Number of hours before session expires (default: 24)
        
        Returns:
            Number of sessions deleted
        
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Get all sessions
                all_sessions = session.query(FeedbackSession).all()
                
                deleted_count = 0
                expiration_time = datetime.now() - timedelta(hours=expiration_hours)
                
                for feedback_session in all_sessions:
                    # Parse created_at timestamp (prefer created_at, fall back to timestamp)
                    if feedback_session.created_at:
                        created_time = datetime.fromisoformat(feedback_session.created_at)
                    elif feedback_session.timestamp:
                        created_time = datetime.fromisoformat(feedback_session.timestamp)
                    else:
                        continue  # Skip if can't determine creation time
                    
                    # Check if expired
                    if created_time < expiration_time:
                        session.delete(feedback_session)
                        deleted_count += 1
                
                session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                else:
                    logger.debug("No expired sessions to clean up")
                
                return deleted_count
                
        except SQLAlchemyError as e:
            logger.error(f"Database error cleaning up expired sessions: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error cleaning up expired sessions: {e}")
            raise

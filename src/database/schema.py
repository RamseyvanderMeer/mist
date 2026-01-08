"""
SQLAlchemy models for MIST database schema.
"""
import json
import numpy as np
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column, Integer, Text, BLOB, REAL, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.sql import func

Base = declarative_base()


class FeedbackSession(Base):
    """Model for feedback_sessions table."""
    __tablename__ = 'feedback_sessions'
    
    session_id = Column(Text, primary_key=True)
    fault_codes = Column(Text)  # JSON array
    obd_data = Column(Text)  # JSON object
    clarification_questions = Column(Text)  # JSON array
    user_responses = Column(Text)  # JSON array
    recommended_guides = Column(Text)  # JSON array
    selected_guide = Column(Text)
    explicit_rating = Column(Integer)  # 1-5
    repair_outcome = Column(Text)  # success/failure/partial
    conversation_corrections = Column(Text)  # JSON array
    timestamp = Column(Text)
    created_at = Column(Text, server_default=func.current_timestamp())
    
    # Relationship to feedback entries
    feedback_entries = relationship("MistFeedback", back_populates="session")
    
    def get_fault_codes(self) -> List[str]:
        """Deserialize fault_codes JSON."""
        if not self.fault_codes:
            return []
        return json.loads(self.fault_codes)
    
    def set_fault_codes(self, codes: List[str]):
        """Serialize fault_codes to JSON."""
        self.fault_codes = json.dumps(codes) if codes else None
    
    def get_obd_data(self) -> Dict[str, Any]:
        """Deserialize obd_data JSON."""
        if not self.obd_data:
            return {}
        return json.loads(self.obd_data)
    
    def set_obd_data(self, data: Dict[str, Any]):
        """Serialize obd_data to JSON."""
        self.obd_data = json.dumps(data) if data else None
    
    def get_clarification_questions(self) -> List[str]:
        """Deserialize clarification_questions JSON."""
        if not self.clarification_questions:
            return []
        return json.loads(self.clarification_questions)
    
    def set_clarification_questions(self, questions: List[str]):
        """Serialize clarification_questions to JSON."""
        self.clarification_questions = json.dumps(questions) if questions else None
    
    def get_user_responses(self) -> List[str]:
        """Deserialize user_responses JSON."""
        if not self.user_responses:
            return []
        return json.loads(self.user_responses)
    
    def set_user_responses(self, responses: List[str]):
        """Serialize user_responses to JSON."""
        self.user_responses = json.dumps(responses) if responses else None
    
    def get_recommended_guides(self) -> List[str]:
        """Deserialize recommended_guides JSON."""
        if not self.recommended_guides:
            return []
        return json.loads(self.recommended_guides)
    
    def set_recommended_guides(self, guides: List[str]):
        """Serialize recommended_guides to JSON."""
        self.recommended_guides = json.dumps(guides) if guides else None
    
    def get_conversation_corrections(self) -> List[Dict[str, Any]]:
        """Deserialize conversation_corrections JSON."""
        if not self.conversation_corrections:
            return []
        return json.loads(self.conversation_corrections)
    
    def set_conversation_corrections(self, corrections: List[Dict[str, Any]]):
        """Serialize conversation_corrections to JSON."""
        self.conversation_corrections = json.dumps(corrections) if corrections else None


class MistEmbedding(Base):
    """Model for mist_embeddings table."""
    __tablename__ = 'mist_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    procedure_id = Column(Text, nullable=False)
    embedding = Column(BLOB)  # 768-dim vector (numpy array)
    embedding_version = Column(Integer)
    created_at = Column(Text, server_default=func.current_timestamp())
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialize embedding BLOB to numpy array."""
        if not self.embedding:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    def set_embedding(self, embedding: np.ndarray):
        """Serialize numpy array to BLOB."""
        if embedding is not None:
            # Ensure float32 and contiguous
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            self.embedding = embedding.tobytes()
        else:
            self.embedding = None
    
    def get_embedding_shape(self) -> Optional[tuple]:
        """Get expected shape of embedding (for validation)."""
        if self.embedding:
            # 768-dim vector
            return (768,)
        return None


class MistFeedback(Base):
    """Model for mist_feedback table."""
    __tablename__ = 'mist_feedback'
    
    feedback_id = Column(Text, primary_key=True)
    session_id = Column(Text, ForeignKey('feedback_sessions.session_id'), nullable=False)
    procedure_id = Column(Text)  # Logical reference to XEP_INFOOBJECTS(ID)
    rating = Column(Integer)  # 1-5
    repair_outcome = Column(Text)  # success/failure/partial
    feedback_text = Column(Text)
    created_at = Column(Text, server_default=func.current_timestamp())
    
    # Relationship to session
    session = relationship("FeedbackSession", back_populates="feedback_entries")


class MistTrainingCheckpoint(Base):
    """Model for mist_training_checkpoints table."""
    __tablename__ = 'mist_training_checkpoints'
    
    checkpoint_id = Column(Text, primary_key=True)
    epoch = Column(Integer)
    loss = Column(REAL)
    validation_loss = Column(REAL)
    embedding_version = Column(Integer)
    checkpoint_path = Column(Text)
    created_at = Column(Text, server_default=func.current_timestamp())

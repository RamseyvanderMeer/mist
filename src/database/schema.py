"""
SQLAlchemy ORM models for MIST database schema.

Matches tables created by scripts/migrations/create_mist_tables.sql.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class FeedbackSession(Base):
    """Stores conversational RAG sessions with fault codes, OBD data, and user responses."""

    __tablename__ = "feedback_sessions"

    session_id = Column(String, primary_key=True)
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
    created_at = Column(Text, default=lambda: datetime.utcnow().isoformat())

    feedback_entries = relationship("MistFeedback", back_populates="session")

    def set_fault_codes(self, codes: List[str]) -> None:
        self.fault_codes = json.dumps(codes) if codes else None

    def get_fault_codes(self) -> List[str]:
        if not self.fault_codes:
            return []
        return json.loads(self.fault_codes)

    def set_obd_data(self, data: Dict[str, Any]) -> None:
        self.obd_data = json.dumps(data) if data else None

    def get_obd_data(self) -> Dict[str, Any]:
        if not self.obd_data:
            return {}
        return json.loads(self.obd_data)

    def set_clarification_questions(self, questions: List[str]) -> None:
        self.clarification_questions = json.dumps(questions) if questions else None

    def get_clarification_questions(self) -> List[str]:
        if not self.clarification_questions:
            return []
        return json.loads(self.clarification_questions)

    def set_user_responses(self, responses: List[str]) -> None:
        self.user_responses = json.dumps(responses) if responses else None

    def get_user_responses(self) -> List[str]:
        if not self.user_responses:
            return []
        return json.loads(self.user_responses)

    def set_recommended_guides(self, guides: List[str]) -> None:
        self.recommended_guides = json.dumps(guides) if guides else None

    def get_recommended_guides(self) -> List[str]:
        if not self.recommended_guides:
            return []
        return json.loads(self.recommended_guides)

    def set_conversation_corrections(self, corrections: List[Dict[str, Any]]) -> None:
        self.conversation_corrections = json.dumps(corrections) if corrections else None

    def get_conversation_corrections(self) -> List[Dict[str, Any]]:
        if not self.conversation_corrections:
            return []
        return json.loads(self.conversation_corrections)


class MistEmbedding(Base):
    """Stores procedure embeddings with versioning support."""

    __tablename__ = "mist_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    procedure_id = Column(String, nullable=False)
    embedding = Column(LargeBinary)  # 768-dim vector (numpy array)
    embedding_version = Column(Integer)
    created_at = Column(Text, default=lambda: datetime.utcnow().isoformat())

    def set_embedding(self, vector: np.ndarray) -> None:
        self.embedding = vector.astype(np.float32).tobytes()

    def get_embedding(self) -> Optional[np.ndarray]:
        if not self.embedding:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)


class MistFeedback(Base):
    """Stores individual feedback entries linked to sessions."""

    __tablename__ = "mist_feedback"

    feedback_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("feedback_sessions.session_id"), nullable=False)
    procedure_id = Column(Text)
    rating = Column(Integer)  # 1-5
    repair_outcome = Column(Text)  # success/failure/partial
    feedback_text = Column(Text)
    created_at = Column(Text, default=lambda: datetime.utcnow().isoformat())

    session = relationship("FeedbackSession", back_populates="feedback_entries")


class MistTrainingCheckpoint(Base):
    """Tracks embedding model training checkpoints."""

    __tablename__ = "mist_training_checkpoints"

    checkpoint_id = Column(String, primary_key=True)
    epoch = Column(Integer)
    loss = Column(Float)
    validation_loss = Column(Float)
    embedding_version = Column(Integer)
    checkpoint_path = Column(Text)
    created_at = Column(Text, default=lambda: datetime.utcnow().isoformat())

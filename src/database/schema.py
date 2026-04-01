"""
SQLAlchemy ORM models for MIST database schema.

Maps to tables created by scripts/migrations/create_mist_tables.sql.
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import Column, Float, ForeignKey, Integer, LargeBinary, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class FeedbackSession(Base):
    """ORM model for feedback_sessions table."""

    __tablename__ = "feedback_sessions"

    session_id = Column(Text, primary_key=True)
    fault_codes = Column(Text)
    obd_data = Column(Text)
    clarification_questions = Column(Text)
    user_responses = Column(Text)
    recommended_guides = Column(Text)
    selected_guide = Column(Text)
    explicit_rating = Column(Integer)
    repair_outcome = Column(Text)
    conversation_corrections = Column(Text)
    timestamp = Column(Text)
    created_at = Column(Text)

    feedback_entries = relationship(
        "MistFeedback", back_populates="session", lazy="select"
    )

    def _set_json(self, attr: str, value) -> None:
        setattr(self, attr, json.dumps(value) if value is not None else None)

    def _get_json(self, attr: str):
        raw = getattr(self, attr)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_fault_codes(self, codes: List[str]) -> None:
        self._set_json("fault_codes", codes)

    def get_fault_codes(self) -> Optional[List[str]]:
        return self._get_json("fault_codes")

    def set_obd_data(self, data: Dict[str, Any]) -> None:
        self._set_json("obd_data", data)

    def get_obd_data(self) -> Optional[Dict[str, Any]]:
        return self._get_json("obd_data")

    def set_clarification_questions(self, questions: List[str]) -> None:
        self._set_json("clarification_questions", questions)

    def get_clarification_questions(self) -> Optional[List[str]]:
        return self._get_json("clarification_questions")

    def set_user_responses(self, responses: List[str]) -> None:
        self._set_json("user_responses", responses)

    def get_user_responses(self) -> Optional[List[str]]:
        return self._get_json("user_responses")

    def set_recommended_guides(self, guides: List[str]) -> None:
        self._set_json("recommended_guides", guides)

    def get_recommended_guides(self) -> Optional[List[str]]:
        return self._get_json("recommended_guides")

    def set_conversation_corrections(self, corrections: List[Dict[str, Any]]) -> None:
        self._set_json("conversation_corrections", corrections)

    def get_conversation_corrections(self) -> Optional[List[Dict[str, Any]]]:
        return self._get_json("conversation_corrections")


class MistEmbedding(Base):
    """ORM model for mist_embeddings table."""

    __tablename__ = "mist_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    procedure_id = Column(Text, nullable=False)
    embedding = Column(LargeBinary)
    embedding_version = Column(Integer)
    created_at = Column(Text)

    def set_embedding(self, vector: np.ndarray) -> None:
        """Serialize a numpy array to bytes for storage."""
        self.embedding = vector.astype(np.float32).tobytes()

    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialize bytes back to a numpy float32 array."""
        if self.embedding is None:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32).copy()


class MistFeedback(Base):
    """ORM model for mist_feedback table."""

    __tablename__ = "mist_feedback"

    feedback_id = Column(Text, primary_key=True)
    session_id = Column(
        Text, ForeignKey("feedback_sessions.session_id"), nullable=False
    )
    procedure_id = Column(Text)
    rating = Column(Integer)
    repair_outcome = Column(Text)
    feedback_text = Column(Text)
    created_at = Column(Text)

    session = relationship("FeedbackSession", back_populates="feedback_entries")


class MistTrainingCheckpoint(Base):
    """ORM model for mist_training_checkpoints table."""

    __tablename__ = "mist_training_checkpoints"

    checkpoint_id = Column(Text, primary_key=True)
    epoch = Column(Integer)
    loss = Column(Float)
    validation_loss = Column(Float)
    embedding_version = Column(Integer)
    checkpoint_path = Column(Text)
    created_at = Column(Text)

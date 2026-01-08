"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Recommendation(BaseModel):
    """Repair guide recommendation"""
    id: str
    title: str
    procedure_name: str
    procedure_id: Optional[str] = None
    score: float
    text: Optional[str] = None


class FaultCodeRequest(BaseModel):
    """Fault code query request"""
    fault_codes: List[str]
    obd_data: Dict
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response"""
    recommendations: List[Recommendation]
    needs_clarification: bool
    clarification_questions: Optional[List[str]] = None
    session_id: str
    query_text: str


class UserResponse(BaseModel):
    """User clarification response"""
    session_id: str
    responses: List[str]


class RatingFeedback(BaseModel):
    """Rating feedback"""
    session_id: str
    rating: int = Field(ge=1, le=5)
    selected_guide: Optional[str] = None


class RepairOutcomeFeedback(BaseModel):
    """Repair outcome feedback"""
    session_id: str
    outcome: str
    details: Optional[Dict] = None


class ConversationCorrection(BaseModel):
    """Conversation correction feedback"""
    session_id: str
    correction: Dict


class FeedbackStatistics(BaseModel):
    """Feedback statistics"""
    total_sessions: int
    rated_sessions: int
    average_rating: float
    repair_outcomes: Dict[str, int]
    corrected_sessions: int
    rating_coverage: float

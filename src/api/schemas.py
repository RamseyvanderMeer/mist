"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class Recommendation(BaseModel):
    """Repair guide recommendation with relevance score."""
    id: str = Field(..., description="Unique identifier for the recommendation")
    title: str = Field(..., description="Title of the repair guide")
    procedure_name: str = Field(..., description="Name of the repair procedure")
    procedure_id: Optional[str] = Field(None, description="Procedure ID from database")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0.0 and 1.0")
    text: Optional[str] = Field(None, description="Optional text content from the repair guide")


class QueryRequest(BaseModel):
    """Query request with fault codes, OBD data, and optional vehicle context."""
    fault_codes: List[str] = Field(..., min_length=1, description="List of fault codes (e.g., ['P0301', 'P0302'])")
    obd_data: Dict[str, Any] = Field(..., description="OBD sensor data dictionary")
    vehicle_context: Optional[Dict[str, Any]] = Field(None, description="Optional vehicle information (model, year, etc.)")
    session_id: Optional[str] = Field(None, description="Optional session ID for continuing existing conversation")


class QueryResponse(BaseModel):
    """Query response with recommendations and optional clarification questions."""
    recommendations: List[Recommendation] = Field(..., description="List of ranked repair guide recommendations")
    needs_clarification: bool = Field(..., description="Whether clarification questions are needed")
    clarification_questions: Optional[List[str]] = Field(None, description="Optional list of clarification questions")
    session_id: str = Field(..., description="Session ID for tracking the conversation")
    query_text: str = Field(..., description="Generated query text from fault codes")


class ClarifyRequest(BaseModel):
    """Clarification request with session ID and user responses."""
    session_id: str = Field(..., min_length=1, description="Session ID for the conversation")
    responses: List[str] = Field(..., min_length=1, description="List of user responses to clarification questions")


class RepairOutcome(str, Enum):
    """Repair outcome enumeration."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class RatingFeedback(BaseModel):
    """Rating feedback for a repair guide recommendation."""
    session_id: str = Field(..., min_length=1, description="Session ID for the conversation")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    selected_guide: Optional[str] = Field(None, description="ID of the guide that was selected")


class RepairOutcomeFeedback(BaseModel):
    """Repair outcome feedback after attempting a repair."""
    session_id: str = Field(..., min_length=1, description="Session ID for the conversation")
    outcome: RepairOutcome = Field(..., description="Repair outcome: success, failure, or partial")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional additional details about the outcome")


class ConversationCorrection(BaseModel):
    """Conversation correction feedback."""
    session_id: str = Field(..., min_length=1, description="Session ID for the conversation")
    correction: Dict[str, Any] = Field(..., description="Correction information dictionary")


class FeedbackStatistics(BaseModel):
    """Feedback statistics summary."""
    total_sessions: int = Field(..., ge=0, description="Total number of sessions")
    rated_sessions: int = Field(..., ge=0, description="Number of sessions with ratings")
    average_rating: float = Field(..., ge=0.0, le=5.0, description="Average rating across all rated sessions")
    repair_outcomes: Dict[str, int] = Field(..., description="Count of repair outcomes by type")
    corrected_sessions: int = Field(..., ge=0, description="Number of sessions with corrections")
    rating_coverage: float = Field(..., ge=0.0, le=1.0, description="Percentage of sessions with ratings")

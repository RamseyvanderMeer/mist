"""
FastAPI server for MIST API endpoints.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    QueryRequest, QueryResponse, ClarifyRequest, RatingFeedback,
    RepairOutcomeFeedback, ConversationCorrection, FeedbackStatistics
)
from src.retrieval.conversational_rag import ConversationalRAG, ConversationalRAGError
from src.feedback.collector import FeedbackCollector
from src.feedback.analyzer import FeedbackAnalyzer
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Optional

logger = logging.getLogger(__name__)

app = FastAPI(title="MIST API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances for lazy initialization
_conversational_rag: Optional[ConversationalRAG] = None
_feedback_collector: Optional[FeedbackCollector] = None
_feedback_analyzer: Optional[FeedbackAnalyzer] = None


def get_conversational_rag() -> ConversationalRAG:
    """Dependency function to get or create ConversationalRAG instance."""
    global _conversational_rag
    if _conversational_rag is None:
        try:
            logger.info("Initializing ConversationalRAG...")
            _conversational_rag = ConversationalRAG()
            logger.info("ConversationalRAG initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConversationalRAG: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize ConversationalRAG: {str(e)}"
            )
    return _conversational_rag


def get_feedback_collector() -> FeedbackCollector:
    """Dependency function to get or create FeedbackCollector instance."""
    global _feedback_collector
    if _feedback_collector is None:
        try:
            logger.info("Initializing FeedbackCollector...")
            _feedback_collector = FeedbackCollector()
            logger.info("FeedbackCollector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FeedbackCollector: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize FeedbackCollector: {str(e)}"
            )
    return _feedback_collector


def get_feedback_analyzer() -> FeedbackAnalyzer:
    """Dependency function to get or create FeedbackAnalyzer instance."""
    global _feedback_analyzer
    if _feedback_analyzer is None:
        try:
            logger.info("Initializing FeedbackAnalyzer...")
            _feedback_analyzer = FeedbackAnalyzer()
            logger.info("FeedbackAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FeedbackAnalyzer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize FeedbackAnalyzer: {str(e)}"
            )
    return _feedback_analyzer


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MIST API"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag: ConversationalRAG = Depends(get_conversational_rag)
):
    """Process fault codes and OBD data"""
    try:
        logger.info(
            f"Processing query: {len(request.fault_codes)} fault codes, "
            f"session_id={request.session_id}"
        )
        
        # Call ConversationalRAG.query()
        result = rag.query(
            fault_codes=request.fault_codes,
            obd_data=request.obd_data or {},
            description=request.description,
            vehicle_context=request.vehicle_context,
            session_id=request.session_id
        )
        
        # Convert response dict to QueryResponse schema
        response = QueryResponse(
            recommendations=result.get("recommendations", []),
            needs_clarification=result.get("needs_clarification", False),
            clarification_questions=result.get("clarification_questions"),
            session_id=result.get("session_id", ""),
            query_text=result.get("query_text", "")
        )
        
        logger.info(
            f"Query completed: session_id={response.session_id}, "
            f"{len(response.recommendations)} recommendations"
        )
        
        return response
        
    except ConversationalRAGError as e:
        logger.error(f"ConversationalRAG error in query endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/clarify", response_model=QueryResponse)
async def clarify(
    request: ClarifyRequest,
    rag: ConversationalRAG = Depends(get_conversational_rag)
):
    """Process clarification responses"""
    try:
        logger.info(
            f"Processing clarification: session_id={request.session_id}, "
            f"{len(request.responses)} responses"
        )
        
        # Call ConversationalRAG.clarify()
        result = rag.clarify(
            session_id=request.session_id,
            responses=request.responses
        )
        
        # Convert response dict to QueryResponse schema
        response = QueryResponse(
            recommendations=result.get("recommendations", []),
            needs_clarification=result.get("needs_clarification", False),
            clarification_questions=result.get("clarification_questions"),
            session_id=result.get("session_id", ""),
            query_text=result.get("query_text", "")
        )
        
        logger.info(
            f"Clarification completed: session_id={response.session_id}, "
            f"{len(response.recommendations)} recommendations"
        )
        
        return response
        
    except ConversationalRAGError as e:
        error_msg = str(e)
        # Check if it's a session not found error
        if "not found" in error_msg.lower() or "Session" in error_msg:
            logger.warning(f"Session not found in clarify endpoint: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )
        else:
            logger.error(f"ConversationalRAG error in clarify endpoint: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Clarification processing failed: {str(e)}"
            )
    except Exception as e:
        logger.error(f"Unexpected error in clarify endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/feedback/rating")
async def submit_rating(
    feedback: RatingFeedback,
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """Submit rating feedback"""
    try:
        logger.info(
            f"Submitting rating feedback: session_id={feedback.session_id}, "
            f"rating={feedback.rating}"
        )
        
        # Update session with rating and selected guide
        # save_session handles both creating new sessions and updating existing ones
        collector.save_session(
            session_id=feedback.session_id,
            explicit_rating=feedback.rating,
            selected_guide=feedback.selected_guide
        )
        
        logger.info(f"Rating feedback saved for session {feedback.session_id}")
        return {"status": "success", "message": "Rating recorded"}
        
    except ValueError as e:
        logger.warning(f"Invalid rating value: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower():
            logger.warning(f"Session not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {feedback.session_id}"
            )
        else:
            logger.error(f"Runtime error saving rating: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"Database error saving rating: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while saving rating feedback"
        )
    except Exception as e:
        logger.error(f"Unexpected error saving rating: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/feedback/outcome")
async def submit_outcome(
    feedback: RepairOutcomeFeedback,
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """Submit repair outcome feedback"""
    try:
        logger.info(
            f"Submitting outcome feedback: session_id={feedback.session_id}, "
            f"outcome={feedback.outcome}"
        )
        
        # Update session with repair outcome
        # Note: details field is not stored in FeedbackSession model
        collector.save_session(
            session_id=feedback.session_id,
            repair_outcome=feedback.outcome.value
        )
        
        logger.info(f"Outcome feedback saved for session {feedback.session_id}")
        return {"status": "success", "message": "Outcome recorded"}
        
    except ValueError as e:
        logger.warning(f"Invalid outcome value: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower():
            logger.warning(f"Session not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {feedback.session_id}"
            )
        else:
            logger.error(f"Runtime error saving outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"Database error saving outcome: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while saving outcome feedback"
        )
    except Exception as e:
        logger.error(f"Unexpected error saving outcome: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/feedback/correction")
async def submit_correction(
    feedback: ConversationCorrection,
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """Submit conversation correction"""
    try:
        logger.info(
            f"Submitting correction feedback: session_id={feedback.session_id}"
        )
        
        # Get existing session to preserve conversation_corrections
        existing_session = collector.get_session(feedback.session_id)
        existing_corrections = []
        if existing_session:
            existing_corrections = existing_session.get("conversation_corrections", [])
        
        # Append new correction to existing corrections
        existing_corrections.append(feedback.correction)
        
        # Update session with conversation corrections
        collector.save_session(
            session_id=feedback.session_id,
            conversation_corrections=existing_corrections
        )
        
        logger.info(f"Correction feedback saved for session {feedback.session_id}")
        return {"status": "success", "message": "Correction recorded"}
        
    except RuntimeError as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower():
            logger.warning(f"Session not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {feedback.session_id}"
            )
        else:
            logger.error(f"Runtime error saving correction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"Database error saving correction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while saving correction feedback"
        )
    except Exception as e:
        logger.error(f"Unexpected error saving correction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/feedback/statistics", response_model=FeedbackStatistics)
async def get_statistics(
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer),
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """Get feedback statistics"""
    try:
        logger.info("Retrieving feedback statistics")
        
        # Get statistics from analyzer
        stats = analyzer.get_statistics()
        
        # Calculate corrected_sessions by querying sessions with conversation_corrections
        # We need to access the database directly for this
        from src.database.connection import create_connection
        from src.database.schema import FeedbackSession
        from src.paths import get_paths
        
        paths = get_paths()
        connection = create_connection(paths.feedback_db)
        
        with connection.session() as session:
            corrected_sessions = session.query(FeedbackSession).filter(
                FeedbackSession.conversation_corrections.isnot(None)
            ).count()
        
        # Map analyzer response to FeedbackStatistics schema
        response = FeedbackStatistics(
            total_sessions=stats.get("total_sessions", 0),
            rated_sessions=stats.get("rated_sessions", 0),
            average_rating=stats.get("average_rating", 0.0),
            repair_outcomes=stats.get("repair_outcomes", {}),
            corrected_sessions=corrected_sessions,
            rating_coverage=stats.get("rating_coverage", 0.0)
        )
        
        logger.info("Statistics retrieved successfully")
        return response
        
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while retrieving statistics"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/feedback/{session_id}")
async def get_feedback_session(
    session_id: str,
    collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """Get feedback session by ID"""
    try:
        logger.info(f"Retrieving feedback session: {session_id}")
        
        session_data = collector.get_session(session_id)
        
        if session_data is None:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        logger.info(f"Session retrieved: {session_id}")
        return session_data
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while retrieving session"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

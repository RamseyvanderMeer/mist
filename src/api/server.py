"""
FastAPI server for MIST API endpoints.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from .schemas import (
    QueryRequest, QueryResponse, ClarifyRequest, RatingFeedback,
    RepairOutcomeFeedback, ConversationCorrection, FeedbackStatistics
)
from src.retrieval.conversational_rag import ConversationalRAG, ConversationalRAGError
from src.feedback.collector import FeedbackCollector
from src.feedback.analyzer import FeedbackAnalyzer
from src.api.security import setup_security
from src.auth.dependencies import (
    get_current_user,
    require_admin,
    tier_limit_for_ratelimit_key,
    limiter as auth_limiter,
)
from src.auth.routes import router as auth_router
from src.database.pg_connection import pg_engine
from src.database.init import init_db
from src.models import Base
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Initialize rate limiter state
limiter = auth_limiter

def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded."""
    limit = None
    reset_hint = None

    detail = getattr(exc, "detail", None)
    if isinstance(detail, str):
        limit = detail
        if "/day" in detail:
            reset_hint = "Try again tomorrow."
        elif "/hour" in detail:
            reset_hint = "Try again later this hour."
        elif "/minute" in detail:
            reset_hint = "Try again in a minute."

    message = f"You have hit your rate limit{f' of {limit}' if limit else ''}."
    if reset_hint:
        message = f"{message} {reset_hint}"

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": message,
            "code": "RATE_LIMIT_EXCEEDED",
            "limit": limit,
            "retry_after": detail if detail is not None else None,
        }
    )

app = FastAPI(
    title="MIST API",
    description="""
    **MIST (Mechanic Intelligence Support Tool) API** provides intelligent repair guide recommendations 
    for BMW vehicles based on fault codes and symptom descriptions.
    
    ## Authentication
    
    Production uses **Google Cloud IAP** (`X-Goog-Iap-Jwt-Assertion`) and/or **Google Sign-In**
    (`Authorization: Bearer <Google ID token>`) when `GOOGLE_OAUTH_CLIENT_IDS` is configured.
    Send `X-Goog-Authenticated-User-Email` matching the token email when using Bearer auth.
    
    ### Headers
    - `X-Goog-Iap-Jwt-Assertion`: IAP assertion JWT (browser/proxy or advanced clients)
    - `Authorization: Bearer`: Google OAuth ID token (mobile / SPA) — requires server env `GOOGLE_OAUTH_CLIENT_IDS`
    - `X-Goog-Authenticated-User-Email` / `X-Goog-Authenticated-User-Id`: identity hints (must match verified token when both are sent)
    
    ### Registration Flow
    1. First-time users must call `POST /auth/register` to create an account
    2. Subsequent requests authenticate via IAP JWT or verified Google ID token
    3. New users are assigned the "blocked" tier by default (no API access)
    4. Contact an admin to upgrade your tier
    
    ## Rate Limiting
    
    Rate limits are tier-based:
    - **blocked**: 0 requests (default for new users)
    - **free**: 10/min, 100/hour, 500/day
    - **premium**: 100/min, 1000/hour, 5000/day
    - **admin**: 1000/min, 10000/hour, 100000/day
    
    ## Key Features
    
    - **Fault Code Search**: Query by OBD-II fault codes (e.g., P0301, 2A87)
    - **Symptom Search**: Describe problems in natural language
    - **Clarification Flow**: Interactive questioning when diagnosis is ambiguous
    - **Feedback Loop**: Submit ratings and outcomes to improve recommendations
    
    ## Typical Usage Flow
    
    1. **Register**: Call `/auth/register` (first time only)
    2. **Submit Query**: Send fault codes and/or symptoms to `/query`
    3. **Review Recommendations**: Get ranked repair guide recommendations
    4. **Clarify (if needed)**: If `needs_clarification=true`, answer questions via `/clarify`
    5. **Submit Feedback**: Rate recommendations and report repair outcomes
    """,
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "MIST Support",
        "email": "support@mist.ai"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

# CORS middleware - restrict in production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include auth routes
app.include_router(auth_router)

# Setup security (rate limiting, API key validation)
api_keys = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else None
setup_security(app, api_keys=api_keys)

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


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and seed data on startup."""
    try:
        # Create tables
        Base.metadata.create_all(bind=pg_engine)
        
        # Seed default data
        from sqlalchemy.orm import Session
        with Session(pg_engine) as db:
            init_db(db)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


@app.get("/health", tags=["Health"], summary="Health Check")
async def health_check():
    """
    Check if the API service is running and healthy.
    
    Returns:
        Service status and name
        
    Example Response:
        ```json
        {
            "status": "healthy",
            "service": "MIST API"
        }
        ```
    """
    return {"status": "healthy", "service": "MIST API"}


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Query Repair Guides",
    responses={
        400: {"description": "Bad Request - No fault codes or description provided"},
        401: {"description": "Unauthorized - Authentication required"},
        403: {"description": "Forbidden - User not registered or tier blocked"},
        429: {"description": "Rate Limit Exceeded"},
        500: {"description": "Internal Server Error - Query processing failed"}
    }
)
@limiter.limit(tier_limit_for_ratelimit_key)
async def query(
    request: Request,
    query_request: QueryRequest,
    rag: ConversationalRAG = Depends(get_conversational_rag),
    current_user = Depends(get_current_user)
):
    """
    Query repair guides by fault codes and/or symptom description.

    This endpoint performs semantic search across 337,000+ BMW repair guides to find
    the most relevant procedures for the given fault codes or symptoms.

    ## Search Types

    **1. Fault Code Search:**
    - Provide OBD-II codes (e.g., P0301, B002A) or BMW hex codes (e.g., 29CC, 2A87)
    - The API automatically converts between formats

    **2. Symptom Search:**
    - Describe problems in natural language (e.g., "engine misfire", "rough idle")
    - Symptom expansion automatically adds related terms

    **3. Combined Search:**
    - Use both fault codes AND description for best results

    ## Response Fields

    - **recommendations**: List of repair guides ranked by relevance (0-1 score)
    - **needs_clarification**: If true, answer questions via `/clarify` endpoint
    - **clarification_questions**: Questions to narrow down diagnosis
    - **session_id**: Save this to continue conversation via `/clarify`
    - **query_text**: The expanded query used for search

    ## Example Request

    ```json
    {
        "fault_codes": ["P0301"],
        "description": "engine misfire cylinder 1",
        "vehicle_context": {"model": "BMW 3-Series", "year": 2019}
    }
    ```

    ## Example Response

    ```json
    {
        "recommendations": [
            {
                "id": "2000014950753",
                "title": "Measure for fault 290900",
                "procedure_name": "Measure for fault 290900",
                "procedure_id": "2000014950753",
                "score": 0.306,
                "text": "Problem/Solution: sporadic fault in the ignition system..."
            }
        ],
        "needs_clarification": false,
        "clarification_questions": null,
        "session_id": "abc-123-def",
        "query_text": "Fault codes: P0301, 29CC. Problem: engine misfire..."
    }
    ```
    """
    # Validate that at least one search criterion is provided
    if not query_request.fault_codes and not query_request.description:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'fault_codes' or 'description' must be provided"
        )
    
    try:
        logger.info(
            f"Processing query for user {current_user.email}: "
            f"{len(query_request.fault_codes)} fault codes, "
            f"session_id={query_request.session_id}"
        )
        
        # Call ConversationalRAG.query()
        result = rag.query(
            fault_codes=query_request.fault_codes,
            obd_data=query_request.obd_data or {},
            description=query_request.description,
            vehicle_context=query_request.vehicle_context,
            session_id=query_request.session_id
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


@app.post(
    "/clarify",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Submit Clarification Responses",
    responses={
        401: {"description": "Unauthorized - Authentication required"},
        403: {"description": "Forbidden - User not registered or tier blocked"},
        404: {"description": "Session not found"},
        429: {"description": "Rate Limit Exceeded"},
        500: {"description": "Processing error"}
    }
)
@limiter.limit(tier_limit_for_ratelimit_key)
async def clarify(
    request: Request,
    clarify_request: ClarifyRequest,
    rag: ConversationalRAG = Depends(get_conversational_rag),
    current_user = Depends(get_current_user)
):
    """
    Submit responses to clarification questions from a previous query.

    Use this endpoint when `needs_clarification=true` in the `/query` response.
    The session_id from the query response must be provided.

    ## When to Use

    1. Call `/query` with fault codes/description
    2. If response has `needs_clarification: true`, display the questions to the user
    3. Collect user answers
    4. Call `/clarify` with the same session_id and user responses
    5. Get refined recommendations

    ## Example Request

    ```json
    {
        "session_id": "abc-123-def",
        "responses": [
            "The engine shakes at idle",
            "Check engine light is on"
        ]
    }
    ```

    ## Example Response

    Same format as `/query` endpoint, with refined recommendations based on clarifications.
    """
    try:
        logger.info(
            f"Processing clarification for user {current_user.email}: "
            f"session_id={clarify_request.session_id}, "
            f"{len(clarify_request.responses)} responses"
        )
        
        # Call ConversationalRAG.clarify()
        result = rag.clarify(
            session_id=clarify_request.session_id,
            responses=clarify_request.responses
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
                detail=f"Session not found: {clarify_request.session_id}"
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


@app.post(
    "/feedback/rating",
    tags=["Feedback"],
    summary="Submit Rating Feedback",
    responses={
        200: {"description": "Rating recorded successfully"},
        400: {"description": "Invalid rating value"},
        401: {"description": "Unauthorized - Authentication required"},
        404: {"description": "Session not found"}
    }
)
async def submit_rating(
    feedback: RatingFeedback,
    collector: FeedbackCollector = Depends(get_feedback_collector),
    current_user = Depends(get_current_user)
):
    """
    Submit a rating (1-5) for the recommendations provided in a session.

    This feedback helps improve future recommendations by tracking which
    suggestions were most helpful to users.

    ## When to Use

    After the user has reviewed the recommendations and selected one,
    submit their satisfaction rating.

    ## Example Request

    ```json
    {
        "session_id": "abc-123-def",
        "rating": 4,
        "selected_guide": "2000014950753"
    }
    ```

    ## Parameters

    - **session_id**: The session ID from the query/clarify response
    - **rating**: Integer from 1 (poor) to 5 (excellent)
    - **selected_guide** (optional): ID of the guide the user selected
    """
    try:
        logger.info(
            f"Submitting rating feedback from user {current_user.email}: "
            f"session_id={feedback.session_id}, rating={feedback.rating}"
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


@app.post(
    "/feedback/outcome",
    tags=["Feedback"],
    summary="Submit Repair Outcome",
    responses={
        200: {"description": "Outcome recorded successfully"},
        400: {"description": "Invalid outcome value"},
        401: {"description": "Unauthorized - Authentication required"},
        404: {"description": "Session not found"}
    }
)
async def submit_outcome(
    feedback: RepairOutcomeFeedback,
    collector: FeedbackCollector = Depends(get_feedback_collector),
    current_user = Depends(get_current_user)
):
    """
    Submit the outcome of the repair attempt (success, failure, or partial).

    This feedback tracks whether the recommended procedure actually fixed
    the problem, helping improve future recommendation accuracy.

    ## When to Use

    After the repair has been attempted, report whether it was successful.

    ## Example Request

    ```json
    {
        "session_id": "abc-123-def",
        "outcome": "success",
        "details": {
            "notes": "Replaced spark plugs, misfire resolved"
        }
    }
    ```

    ## Outcome Values

    - **success**: The repair completely fixed the problem
    - **failure**: The repair did not fix the problem
    - **partial**: The repair helped but didn't fully resolve the issue

    ## Parameters

    - **session_id**: The session ID from the query/clarify response
    - **outcome**: One of "success", "failure", or "partial"
    - **details** (optional): Additional information about the outcome
    """
    try:
        logger.info(
            f"Submitting outcome feedback from user {current_user.email}: "
            f"session_id={feedback.session_id}, outcome={feedback.outcome}"
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


@app.post(
    "/feedback/correction",
    tags=["Feedback"],
    summary="Submit Conversation Correction",
    responses={
        200: {"description": "Correction recorded successfully"},
        401: {"description": "Unauthorized - Authentication required"},
        404: {"description": "Session not found"}
    }
)
async def submit_correction(
    feedback: ConversationCorrection,
    collector: FeedbackCollector = Depends(get_feedback_collector),
    current_user = Depends(get_current_user)
):
    """
    Submit a correction to the conversation or recommendations.

    Use this endpoint when the recommended procedures were incorrect
    and you want to provide feedback about what the correct approach should be.

    ## When to Use

    When the repair failed and you want to document what the actual
    correct procedure was for future improvement.

    ## Example Request

    ```json
    {
        "session_id": "abc-123-def",
        "correction": {
            "issue": "Wrong procedure recommended",
            "actual_fault": "Ignition coil failure",
            "correct_procedure": "Replace ignition coil",
            "notes": "The misfire was caused by faulty ignition coil, not spark plugs"
        }
    }
    ```

    ## Parameters

    - **session_id**: The session ID from the query/clarify response
    - **correction**: A dictionary containing correction information.
      Can include any relevant fields such as:
      - issue: Description of what was wrong
      - actual_fault: The actual problem found
      - correct_procedure: What should have been recommended
      - notes: Any additional context
    """
    try:
        logger.info(
            f"Submitting correction feedback from user {current_user.email}: "
            f"session_id={feedback.session_id}"
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


@app.get(
    "/feedback/statistics",
    response_model=FeedbackStatistics,
    tags=["Feedback"],
    summary="Get Feedback Statistics",
    responses={
        401: {"description": "Unauthorized - Authentication required"},
        403: {"description": "Forbidden - Admin access required"}
    }
)
async def get_statistics(
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer),
    collector: FeedbackCollector = Depends(get_feedback_collector),
    admin_user = Depends(require_admin)
):
    """
    Get aggregate statistics about feedback submissions.

    Returns summary metrics about user ratings, repair outcomes,
    and overall system performance.

    ## Response Fields

    - **total_sessions**: Total number of diagnostic sessions
    - **rated_sessions**: Number of sessions with explicit ratings
    - **average_rating**: Average rating across all rated sessions (0-5)
    - **repair_outcomes**: Count of success/failure/partial outcomes
    - **corrected_sessions**: Number of sessions with user corrections
    - **rating_coverage**: Percentage of sessions that received ratings

    ## Example Response

    ```json
    {
        "total_sessions": 150,
        "rated_sessions": 120,
        "average_rating": 4.2,
        "repair_outcomes": {
            "success": 85,
            "failure": 20,
            "partial": 15
        },
        "corrected_sessions": 10,
        "rating_coverage": 0.8
    }
    ```
    """
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


@app.get(
    "/feedback/{session_id}",
    tags=["Feedback"],
    summary="Get Session Feedback",
    responses={
        200: {"description": "Session feedback data"},
        401: {"description": "Unauthorized - Authentication required"},
        403: {"description": "Forbidden - Admin access required"},
        404: {"description": "Session not found"}
    }
)
async def get_feedback_session(
    session_id: str,
    collector: FeedbackCollector = Depends(get_feedback_collector),
    admin_user = Depends(require_admin)
):
    """
    Retrieve all feedback data for a specific session.

    Returns the complete feedback record including ratings,
    outcomes, and corrections submitted for this session.

    ## Parameters

    - **session_id**: The session ID from query/clarify response

    ## Response Fields

    - **session_id**: Session identifier
    - **fault_codes**: Fault codes from the query
    - **explicit_rating**: User rating (1-5) if submitted
    - **repair_outcome**: Outcome (success/failure/partial) if submitted
    - **conversation_corrections**: List of corrections if any
    - **timestamp**: When the session was created

    ## Example Response

    ```json
    {
        "session_id": "abc-123-def",
        "fault_codes": ["P0301"],
        "explicit_rating": 4,
        "repair_outcome": "success",
        "conversation_corrections": null,
        "timestamp": "2026-03-31T06:57:15.693842"
    }
    ```
    """
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

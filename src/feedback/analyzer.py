"""
Feedback analysis and statistics.

Uses SQLAlchemy ORM models for database operations.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

from sqlalchemy import func, case
from sqlalchemy.exc import SQLAlchemyError

from ..database.connection import create_connection
from ..database.migrations import init_database
from ..database.schema import FeedbackSession, MistFeedback
from ..paths import get_paths

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """
    Analyzes feedback data for insights and trends.
    
    Uses SQLAlchemy ORM models for database operations with proper
    session management and error handling.
    """
    
    def __init__(self, db_path: str | None = None):
        """
        Initialize feedback analyzer.
        
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
        
        logger.info(f"Initialized FeedbackAnalyzer with database: {self.db_path}")
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall feedback statistics.
        
        Returns:
            Dictionary containing:
            - total_sessions: Total number of feedback sessions
            - rated_sessions: Number of sessions with explicit ratings
            - average_rating: Average rating (1-5 scale)
            - repair_outcomes: Dictionary mapping outcome to count
            - rating_coverage: Percentage of sessions with ratings
            - total_feedback_entries: Total number of feedback entries
            - procedure_coverage: Number of unique procedures with feedback
        
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with self._connection.session() as session:
                # Total sessions
                total_sessions = session.query(FeedbackSession).count()
                
                # Rated sessions (from FeedbackSession.explicit_rating)
                rated_sessions = session.query(FeedbackSession).filter(
                    FeedbackSession.explicit_rating.isnot(None)
                ).count()
                
                # Average rating (from FeedbackSession.explicit_rating)
                avg_rating_result = session.query(
                    func.avg(FeedbackSession.explicit_rating)
                ).filter(
                    FeedbackSession.explicit_rating.isnot(None)
                ).scalar()
                avg_rating = float(avg_rating_result) if avg_rating_result is not None else 0.0
                
                # Repair outcomes from FeedbackSession
                outcome_counts = session.query(
                    FeedbackSession.repair_outcome,
                    func.count(FeedbackSession.session_id)
                ).filter(
                    FeedbackSession.repair_outcome.isnot(None)
                ).group_by(FeedbackSession.repair_outcome).all()
                
                repair_outcomes = {outcome: count for outcome, count in outcome_counts}
                
                # Total feedback entries
                total_feedback_entries = session.query(MistFeedback).count()
                
                # Procedure coverage (unique procedures with feedback)
                procedure_coverage = session.query(
                    func.count(func.distinct(MistFeedback.procedure_id))
                ).filter(
                    MistFeedback.procedure_id.isnot(None)
                ).scalar() or 0
                
                rating_coverage = rated_sessions / total_sessions if total_sessions > 0 else 0.0
                
                return {
                    "total_sessions": total_sessions,
                    "rated_sessions": rated_sessions,
                    "average_rating": avg_rating,
                    "repair_outcomes": repair_outcomes,
                    "rating_coverage": rating_coverage,
                    "total_feedback_entries": total_feedback_entries,
                    "procedure_coverage": procedure_coverage,
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting statistics: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting statistics: {e}")
            raise
    
    def get_procedure_ratings(
        self,
        min_rating_threshold: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """
        Get ratings and statistics for each procedure.
        
        Calculates per-procedure statistics including average rating, rating count,
        outcome distribution, and combined score.
        
        Args:
            min_rating_threshold: Optional threshold (1-5) to filter low-rated procedures.
                                 If provided, only returns procedures with average rating
                                 below this threshold.
        
        Returns:
            List of dictionaries, each containing:
            - procedure_id: Procedure identifier
            - average_rating: Average rating (1-5 scale)
            - rating_count: Number of ratings for this procedure
            - success_count: Number of success outcomes
            - partial_count: Number of partial outcomes
            - failure_count: Number of failure outcomes
            - combined_score: Combined score (0.0-1.0) from ratings and outcomes
            - total_feedback: Total number of feedback entries for this procedure
        
        Raises:
            ValueError: If min_rating_threshold is not in valid range (1-5)
            SQLAlchemyError: If database operation fails
        """
        if min_rating_threshold is not None and not (1 <= min_rating_threshold <= 5):
            raise ValueError(
                f"min_rating_threshold must be between 1 and 5, got {min_rating_threshold}"
            )
        
        try:
            with self._connection.session() as session:
                # Get all feedback entries grouped by procedure_id
                procedure_feedback = session.query(
                    MistFeedback.procedure_id,
                    func.avg(MistFeedback.rating).label('avg_rating'),
                    func.count(MistFeedback.feedback_id).label('total_feedback'),
                    func.sum(
                        case((MistFeedback.rating.isnot(None), 1), else_=0)
                    ).label('rating_count'),
                    func.sum(
                        case((MistFeedback.repair_outcome == 'success', 1), else_=0)
                    ).label('success_count'),
                    func.sum(
                        case((MistFeedback.repair_outcome == 'partial', 1), else_=0)
                    ).label('partial_count'),
                    func.sum(
                        case((MistFeedback.repair_outcome == 'failure', 1), else_=0)
                    ).label('failure_count'),
                ).filter(
                    MistFeedback.procedure_id.isnot(None)
                ).group_by(MistFeedback.procedure_id).all()
                
                results = []
                
                for row in procedure_feedback:
                    procedure_id = row.procedure_id
                    avg_rating = float(row.avg_rating) if row.avg_rating is not None else None
                    rating_count = row.rating_count or 0
                    success_count = row.success_count or 0
                    partial_count = row.partial_count or 0
                    failure_count = row.failure_count or 0
                    total_feedback = row.total_feedback or 0
                    
                    # Calculate combined score (reuse logic from FeedbackCollector.get_procedure_score)
                    combined_score = None
                    
                    if avg_rating is not None:
                        # Normalize from 1-5 scale to 0.0-1.0
                        normalized_rating = (avg_rating - 1) / 4.0
                    else:
                        normalized_rating = None
                    
                    # Calculate success rate
                    total_outcomes = success_count + partial_count + failure_count
                    if total_outcomes > 0:
                        # Weight: success=1.0, partial=0.5, failure=0.0
                        success_rate = (success_count + 0.5 * partial_count) / total_outcomes
                    else:
                        success_rate = None
                    
                    # Combine scores (weighted average)
                    # If both available: 60% rating, 40% outcome
                    # If only one available: use that
                    if normalized_rating is not None and success_rate is not None:
                        combined_score = 0.6 * normalized_rating + 0.4 * success_rate
                    elif normalized_rating is not None:
                        combined_score = normalized_rating
                    elif success_rate is not None:
                        combined_score = success_rate
                    
                    # Ensure score is in valid range
                    if combined_score is not None:
                        combined_score = max(0.0, min(1.0, combined_score))
                    
                    # Filter by threshold if provided
                    if min_rating_threshold is not None:
                        if avg_rating is None or avg_rating >= min_rating_threshold:
                            continue
                    
                    results.append({
                        "procedure_id": procedure_id,
                        "average_rating": avg_rating,
                        "rating_count": rating_count,
                        "success_count": success_count,
                        "partial_count": partial_count,
                        "failure_count": failure_count,
                        "combined_score": combined_score,
                        "total_feedback": total_feedback,
                    })
                
                # Sort by average_rating (lowest first) or combined_score
                results.sort(key=lambda x: (
                    x["average_rating"] if x["average_rating"] is not None else float('inf'),
                    x["combined_score"] if x["combined_score"] is not None else float('inf')
                ))
                
                return results
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting procedure ratings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting procedure ratings: {e}")
            raise
    
    def get_trends(
        self,
        granularity: str = "day"
    ) -> dict[str, dict[str, Any]]:
        """
        Get feedback trends over time.
        
        Tracks feedback metrics over time periods with configurable granularity.
        
        Args:
            granularity: Time granularity for grouping. One of: 'day', 'week', 'month'.
                        Default: 'day'
        
        Returns:
            Dictionary mapping time period strings (ISO format dates) to dictionaries containing:
            - period: Time period identifier (ISO date string)
            - session_count: Number of sessions in this period
            - average_rating: Average rating for this period
            - rated_sessions: Number of rated sessions
            - success_count: Number of success outcomes
            - partial_count: Number of partial outcomes
            - failure_count: Number of failure outcomes
            - feedback_count: Number of feedback entries
        
        Raises:
            ValueError: If granularity is not one of the supported values
            SQLAlchemyError: If database operation fails
        """
        valid_granularities = {"day", "week", "month"}
        if granularity not in valid_granularities:
            raise ValueError(
                f"granularity must be one of {valid_granularities}, got {granularity}"
            )
        
        try:
            with self._connection.session() as session:
                # Get all sessions with timestamps
                sessions = session.query(
                    FeedbackSession.timestamp,
                    FeedbackSession.created_at,
                    FeedbackSession.explicit_rating,
                    FeedbackSession.repair_outcome,
                ).filter(
                    (FeedbackSession.timestamp.isnot(None)) | (FeedbackSession.created_at.isnot(None))
                ).all()
                
                # Get all feedback entries with timestamps
                feedback_entries = session.query(
                    MistFeedback.created_at,
                    MistFeedback.repair_outcome,
                ).filter(
                    MistFeedback.created_at.isnot(None)
                ).all()
                
                # Group by time period
                trends: dict[str, dict[str, Any]] = defaultdict(lambda: {
                    "period": "",
                    "session_count": 0,
                    "average_rating": None,
                    "rated_sessions": 0,
                    "rating_sum": 0.0,
                    "success_count": 0,
                    "partial_count": 0,
                    "failure_count": 0,
                    "feedback_count": 0,
                })
                
                # Process sessions
                for session_row in sessions:
                    # Use timestamp if available, otherwise created_at
                    timestamp_str = session_row.timestamp or session_row.created_at
                    if not timestamp_str:
                        continue
                    
                    try:
                        # Parse ISO format timestamp
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Remove timezone for grouping
                        if dt.tzinfo:
                            dt = dt.replace(tzinfo=None)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
                        continue
                    
                    # Group by granularity
                    if granularity == "day":
                        period_key = dt.strftime("%Y-%m-%d")
                    elif granularity == "week":
                        # Get Monday of the week
                        monday = dt - timedelta(days=dt.weekday())
                        period_key = monday.strftime("%Y-%m-%d")
                    else:  # month
                        period_key = dt.strftime("%Y-%m")
                    
                    trend = trends[period_key]
                    trend["period"] = period_key
                    trend["session_count"] += 1
                    
                    # Track ratings
                    if session_row.explicit_rating is not None:
                        trend["rated_sessions"] += 1
                        trend["rating_sum"] += float(session_row.explicit_rating)
                    
                    # Track outcomes
                    if session_row.repair_outcome == "success":
                        trend["success_count"] += 1
                    elif session_row.repair_outcome == "partial":
                        trend["partial_count"] += 1
                    elif session_row.repair_outcome == "failure":
                        trend["failure_count"] += 1
                
                # Process feedback entries
                for feedback_row in feedback_entries:
                    timestamp_str = feedback_row.created_at
                    if not timestamp_str:
                        continue
                    
                    try:
                        # Parse ISO format timestamp
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Remove timezone for grouping
                        if dt.tzinfo:
                            dt = dt.replace(tzinfo=None)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
                        continue
                    
                    # Group by granularity
                    if granularity == "day":
                        period_key = dt.strftime("%Y-%m-%d")
                    elif granularity == "week":
                        # Get Monday of the week
                        monday = dt - timedelta(days=dt.weekday())
                        period_key = monday.strftime("%Y-%m-%d")
                    else:  # month
                        period_key = dt.strftime("%Y-%m")
                    
                    if period_key in trends:
                        trends[period_key]["feedback_count"] += 1
                        
                        # Track feedback outcomes
                        if feedback_row.repair_outcome == "success":
                            trends[period_key]["success_count"] += 1
                        elif feedback_row.repair_outcome == "partial":
                            trends[period_key]["partial_count"] += 1
                        elif feedback_row.repair_outcome == "failure":
                            trends[period_key]["failure_count"] += 1
                
                # Calculate average ratings and finalize
                result: dict[str, dict[str, Any]] = {}
                for period_key, trend in sorted(trends.items()):
                    avg_rating = None
                    if trend["rated_sessions"] > 0:
                        avg_rating = trend["rating_sum"] / trend["rated_sessions"]
                    
                    result[period_key] = {
                        "period": period_key,
                        "session_count": trend["session_count"],
                        "average_rating": float(avg_rating) if avg_rating is not None else None,
                        "rated_sessions": trend["rated_sessions"],
                        "success_count": trend["success_count"],
                        "partial_count": trend["partial_count"],
                        "failure_count": trend["failure_count"],
                        "feedback_count": trend["feedback_count"],
                    }
                
                return result
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting trends: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting trends: {e}")
            raise

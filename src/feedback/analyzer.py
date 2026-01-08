"""
Feedback analysis and statistics.
"""
import sqlite3
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """Analyzes feedback data for insights"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM feedback_sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Rated sessions
        cursor.execute("SELECT COUNT(*) FROM feedback_sessions WHERE explicit_rating IS NOT NULL")
        rated_sessions = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute("SELECT AVG(explicit_rating) FROM feedback_sessions WHERE explicit_rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0] or 0.0
        
        # Repair outcomes
        cursor.execute("SELECT repair_outcome, COUNT(*) FROM feedback_sessions WHERE repair_outcome IS NOT NULL GROUP BY repair_outcome")
        outcomes = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "rated_sessions": rated_sessions,
            "average_rating": float(avg_rating),
            "repair_outcomes": outcomes,
            "rating_coverage": rated_sessions / total_sessions if total_sessions > 0 else 0.0
        }

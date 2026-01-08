"""
Feedback collection and storage system.
"""
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and stores user feedback for self-improvement.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize feedback collector.
        
        Args:
            db_path: Path to feedback database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize feedback database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_sessions (
                session_id TEXT PRIMARY KEY,
                fault_codes TEXT,
                obd_data TEXT,
                clarification_questions TEXT,
                user_responses TEXT,
                recommended_guides TEXT,
                selected_guide TEXT,
                explicit_rating INTEGER,
                repair_outcome TEXT,
                conversation_corrections TEXT,
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT,
                event_type TEXT,
                event_data TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, fault_codes: List[str], obd_data: Dict) -> str:
        """Create new feedback session"""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback_sessions (session_id, fault_codes, obd_data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, json.dumps(fault_codes), json.dumps(obd_data), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return session_id
    
    def add_rating(self, session_id: str, rating: int, selected_guide: Optional[str] = None):
        """Add explicit rating feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE feedback_sessions
            SET explicit_rating = ?, selected_guide = ?
            WHERE session_id = ?
        """, (rating, selected_guide, session_id))
        
        conn.commit()
        conn.close()
    
    def add_repair_outcome(self, session_id: str, outcome: str, details: Optional[Dict] = None):
        """Add repair outcome feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE feedback_sessions
            SET repair_outcome = ?
            WHERE session_id = ?
        """, (outcome, session_id))
        
        conn.commit()
        conn.close()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM feedback_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "session_id": row[0],
            "fault_codes": json.loads(row[1] or "[]"),
            "obd_data": json.loads(row[2] or "{}"),
            "explicit_rating": row[7],
            "repair_outcome": row[8]
        }

#!/usr/bin/env python3
"""
Utility script for collecting and managing feedback data.

Provides CLI interface for:
- Adding feedback (sessions and individual entries)
- Viewing statistics
- Exporting feedback data (JSON/CSV)
- Listing sessions and feedback entries
"""
import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# Add workspace root to path so src can be imported as a package
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from src.feedback.collector import FeedbackCollector
from src.feedback.analyzer import FeedbackAnalyzer
from src.paths import get_paths
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _parse_json_input(json_input: str) -> dict[str, Any]:
    """
    Parse JSON from string or file path.
    
    Args:
        json_input: JSON string or path to JSON file
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON is invalid or file doesn't exist
    """
    # Try as file path first
    json_path = Path(json_input)
    if json_path.exists() and json_path.is_file():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {json_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file {json_path}: {e}")
    
    # Try as JSON string
    try:
        return json.loads(json_input)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def _format_statistics(stats: dict[str, Any], format_type: str = "table") -> str:
    """
    Format statistics for display.
    
    Args:
        stats: Statistics dictionary
        format_type: Output format ('table' or 'json')
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(stats, indent=2)
    
    # Table format
    lines = []
    lines.append("Feedback Statistics:")
    lines.append(f"  Total sessions: {stats['total_sessions']}")
    lines.append(f"  Rated sessions: {stats['rated_sessions']}")
    lines.append(f"  Average rating: {stats['average_rating']:.2f}")
    lines.append(f"  Rating coverage: {stats['rating_coverage']:.1%}")
    lines.append(f"  Total feedback entries: {stats['total_feedback_entries']}")
    lines.append(f"  Procedure coverage: {stats['procedure_coverage']}")
    lines.append(f"  Repair outcomes: {stats['repair_outcomes']}")
    
    return "\n".join(lines)


def _export_to_json(
    collector: FeedbackCollector,
    analyzer: FeedbackAnalyzer,
    output_path: Path,
    sessions_only: bool = False,
    feedback_only: bool = False,
    procedure_id: Optional[str] = None,
    since: Optional[str] = None
) -> None:
    """
    Export feedback data to JSON format.
    
    Args:
        collector: FeedbackCollector instance
        analyzer: FeedbackAnalyzer instance
        output_path: Output file path
        sessions_only: Export only sessions
        feedback_only: Export only feedback entries
        procedure_id: Filter by procedure ID
        since: Filter by date (ISO format)
    """
    from src.database.schema import FeedbackSession, MistFeedback
    
    output_data = {}
    
    if not feedback_only:
        # Export sessions
        with collector._connection.session() as session:
            query = session.query(FeedbackSession)
            
            if since:
                try:
                    since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    if since_dt.tzinfo:
                        since_dt = since_dt.replace(tzinfo=None)
                    since_str = since_dt.isoformat()
                    query = query.filter(
                        (FeedbackSession.timestamp >= since_str) |
                        (FeedbackSession.created_at >= since_str)
                    )
                except ValueError as e:
                    logger.warning(f"Invalid date format '{since}': {e}")
            
            sessions = query.all()
            sessions_data = []
            
            for fb_session in sessions:
                session_dict = {
                    "session_id": fb_session.session_id,
                    "fault_codes": fb_session.get_fault_codes(),
                    "obd_data": fb_session.get_obd_data(),
                    "clarification_questions": fb_session.get_clarification_questions(),
                    "user_responses": fb_session.get_user_responses(),
                    "recommended_guides": fb_session.get_recommended_guides(),
                    "selected_guide": fb_session.selected_guide,
                    "explicit_rating": fb_session.explicit_rating,
                    "repair_outcome": fb_session.repair_outcome,
                    "conversation_corrections": fb_session.get_conversation_corrections(),
                    "timestamp": fb_session.timestamp,
                    "created_at": fb_session.created_at,
                }
                
                # Include feedback entries if not sessions_only
                if not sessions_only:
                    feedback_entries = [
                        {
                            "feedback_id": entry.feedback_id,
                            "procedure_id": entry.procedure_id,
                            "rating": entry.rating,
                            "repair_outcome": entry.repair_outcome,
                            "feedback_text": entry.feedback_text,
                            "created_at": entry.created_at,
                        }
                        for entry in fb_session.feedback_entries
                        if procedure_id is None or entry.procedure_id == procedure_id
                    ]
                    session_dict["feedback_entries"] = feedback_entries
                
                sessions_data.append(session_dict)
            
            output_data["sessions"] = sessions_data
    
    if not sessions_only:
        # Export feedback entries separately if needed
        with collector._connection.session() as session:
            query = session.query(MistFeedback)
            
            if procedure_id:
                query = query.filter(MistFeedback.procedure_id == procedure_id)
            
            if since:
                try:
                    since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    if since_dt.tzinfo:
                        since_dt = since_dt.replace(tzinfo=None)
                    since_str = since_dt.isoformat()
                    query = query.filter(MistFeedback.created_at >= since_str)
                except ValueError as e:
                    logger.warning(f"Invalid date format '{since}': {e}")
            
            feedback_entries = query.all()
            feedback_data = [
                {
                    "feedback_id": entry.feedback_id,
                    "session_id": entry.session_id,
                    "procedure_id": entry.procedure_id,
                    "rating": entry.rating,
                    "repair_outcome": entry.repair_outcome,
                    "feedback_text": entry.feedback_text,
                    "created_at": entry.created_at,
                }
                for entry in feedback_entries
            ]
            
            if feedback_only or not output_data.get("sessions"):
                output_data["feedback_entries"] = feedback_data
            # If sessions are included, feedback is already nested in sessions
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Exported data to {output_path}")


def _export_to_csv(
    collector: FeedbackCollector,
    analyzer: FeedbackAnalyzer,
    output_path: Path,
    sessions_only: bool = False,
    feedback_only: bool = False,
    procedure_id: Optional[str] = None,
    since: Optional[str] = None
) -> None:
    """
    Export feedback data to CSV format.
    
    Args:
        collector: FeedbackCollector instance
        analyzer: FeedbackAnalyzer instance
        output_path: Output file path (base name, will create multiple files)
        sessions_only: Export only sessions
        feedback_only: Export only feedback entries
        procedure_id: Filter by procedure ID
        since: Filter by date (ISO format)
    """
    from src.database.schema import FeedbackSession, MistFeedback
    
    if not feedback_only:
        # Export sessions
        sessions_path = output_path.with_suffix('.sessions.csv')
        with collector._connection.session() as session:
            query = session.query(FeedbackSession)
            
            if since:
                try:
                    since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    if since_dt.tzinfo:
                        since_dt = since_dt.replace(tzinfo=None)
                    since_str = since_dt.isoformat()
                    query = query.filter(
                        (FeedbackSession.timestamp >= since_str) |
                        (FeedbackSession.created_at >= since_str)
                    )
                except ValueError as e:
                    logger.warning(f"Invalid date format '{since}': {e}")
            
            sessions = query.all()
            
            if sessions:
                sessions_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sessions_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'session_id', 'fault_codes', 'obd_data', 'clarification_questions',
                        'user_responses', 'recommended_guides', 'selected_guide',
                        'explicit_rating', 'repair_outcome', 'conversation_corrections',
                        'timestamp', 'created_at'
                    ])
                    writer.writeheader()
                    
                    for fb_session in sessions:
                        writer.writerow({
                            'session_id': fb_session.session_id,
                            'fault_codes': json.dumps(fb_session.get_fault_codes()),
                            'obd_data': json.dumps(fb_session.get_obd_data()),
                            'clarification_questions': json.dumps(fb_session.get_clarification_questions()),
                            'user_responses': json.dumps(fb_session.get_user_responses()),
                            'recommended_guides': json.dumps(fb_session.get_recommended_guides()),
                            'selected_guide': fb_session.selected_guide,
                            'explicit_rating': fb_session.explicit_rating,
                            'repair_outcome': fb_session.repair_outcome,
                            'conversation_corrections': json.dumps(fb_session.get_conversation_corrections()),
                            'timestamp': fb_session.timestamp,
                            'created_at': fb_session.created_at,
                        })
                
                logger.info(f"Exported sessions to {sessions_path}")
    
    if not sessions_only:
        # Export feedback entries
        feedback_path = output_path.with_suffix('.feedback.csv')
        with collector._connection.session() as session:
            query = session.query(MistFeedback)
            
            if procedure_id:
                query = query.filter(MistFeedback.procedure_id == procedure_id)
            
            if since:
                try:
                    since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    if since_dt.tzinfo:
                        since_dt = since_dt.replace(tzinfo=None)
                    since_str = since_dt.isoformat()
                    query = query.filter(MistFeedback.created_at >= since_str)
                except ValueError as e:
                    logger.warning(f"Invalid date format '{since}': {e}")
            
            feedback_entries = query.all()
            
            if feedback_entries:
                feedback_path.parent.mkdir(parents=True, exist_ok=True)
                with open(feedback_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'feedback_id', 'session_id', 'procedure_id', 'rating',
                        'repair_outcome', 'feedback_text', 'created_at'
                    ])
                    writer.writeheader()
                    
                    for entry in feedback_entries:
                        writer.writerow({
                            'feedback_id': entry.feedback_id,
                            'session_id': entry.session_id,
                            'procedure_id': entry.procedure_id,
                            'rating': entry.rating,
                            'repair_outcome': entry.repair_outcome,
                            'feedback_text': entry.feedback_text,
                            'created_at': entry.created_at,
                        })
                
                logger.info(f"Exported feedback entries to {feedback_path}")


def _list_sessions(
    collector: FeedbackCollector,
    session_id: Optional[str] = None,
    limit: Optional[int] = None
) -> None:
    """
    List feedback sessions.
    
    Args:
        collector: FeedbackCollector instance
        session_id: Filter by session ID
        limit: Limit number of results
    """
    from database.schema import FeedbackSession
    
    with collector._connection.session() as session:
        query = session.query(FeedbackSession)
        
        if session_id:
            query = query.filter(FeedbackSession.session_id == session_id)
        
        query = query.order_by(FeedbackSession.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        sessions = query.all()
        
        if not sessions:
            print("No sessions found.")
            return
        
        print(f"\nFound {len(sessions)} session(s):\n")
        for fb_session in sessions:
            print(f"Session ID: {fb_session.session_id}")
            print(f"  Created: {fb_session.created_at}")
            print(f"  Timestamp: {fb_session.timestamp or 'N/A'}")
            print(f"  Fault codes: {fb_session.get_fault_codes()}")
            print(f"  Rating: {fb_session.explicit_rating or 'N/A'}")
            print(f"  Outcome: {fb_session.repair_outcome or 'N/A'}")
            print(f"  Selected guide: {fb_session.selected_guide or 'N/A'}")
            print(f"  Feedback entries: {len(fb_session.feedback_entries)}")
            print()


def _list_feedback(
    collector: FeedbackCollector,
    session_id: Optional[str] = None,
    procedure_id: Optional[str] = None,
    limit: Optional[int] = None
) -> None:
    """
    List feedback entries.
    
    Args:
        collector: FeedbackCollector instance
        session_id: Filter by session ID
        procedure_id: Filter by procedure ID
        limit: Limit number of results
    """
    from database.schema import MistFeedback
    
    with collector._connection.session() as session:
        query = session.query(MistFeedback)
        
        if session_id:
            query = query.filter(MistFeedback.session_id == session_id)
        
        if procedure_id:
            query = query.filter(MistFeedback.procedure_id == procedure_id)
        
        query = query.order_by(MistFeedback.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        entries = query.all()
        
        if not entries:
            print("No feedback entries found.")
            return
        
        print(f"\nFound {len(entries)} feedback entry/entries:\n")
        for entry in entries:
            print(f"Feedback ID: {entry.feedback_id}")
            print(f"  Session ID: {entry.session_id}")
            print(f"  Procedure ID: {entry.procedure_id or 'N/A'}")
            print(f"  Rating: {entry.rating or 'N/A'}")
            print(f"  Outcome: {entry.repair_outcome or 'N/A'}")
            print(f"  Text: {entry.feedback_text or 'N/A'}")
            print(f"  Created: {entry.created_at}")
            print()


def cmd_add(args: argparse.Namespace, collector: FeedbackCollector) -> int:
    """Handle add subcommand."""
    try:
        # Parse fault codes
        fault_codes = None
        if args.fault_codes:
            fault_codes = [code.strip() for code in args.fault_codes.split(',')]
        
        # Parse OBD data
        obd_data = None
        if args.obd_data:
            obd_data = _parse_json_input(args.obd_data)
        
        # Validate rating
        if args.rating is not None:
            if not (1 <= args.rating <= 5):
                logger.error("Rating must be between 1 and 5")
                return 1
        
        # Validate outcome
        if args.outcome is not None:
            valid_outcomes = {"success", "failure", "partial"}
            if args.outcome not in valid_outcomes:
                logger.error(f"Outcome must be one of {valid_outcomes}")
                return 1
        
        # Determine if this is session feedback or individual feedback
        if args.procedure_id:
            # Individual feedback entry
            if not args.session_id:
                logger.error("--session-id is required when adding individual feedback")
                return 1
            
            feedback_id = collector.save_feedback(
                session_id=args.session_id,
                procedure_id=args.procedure_id,
                rating=args.rating,
                repair_outcome=args.outcome,
                feedback_text=args.feedback_text
            )
            print(f"Saved feedback entry: {feedback_id}")
        else:
            # Session feedback
            session_id = collector.save_session(
                session_id=args.session_id,
                fault_codes=fault_codes,
                obd_data=obd_data,
                explicit_rating=args.rating,
                repair_outcome=args.outcome,
                selected_guide=args.selected_guide
            )
            print(f"Saved session: {session_id}")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        return 1


def cmd_stats(args: argparse.Namespace, collector: FeedbackCollector, analyzer: FeedbackAnalyzer) -> int:
    """Handle stats subcommand."""
    try:
        if args.procedures:
            # Per-procedure statistics
            procedure_stats = analyzer.get_procedure_ratings(
                min_rating_threshold=args.min_rating_threshold
            )
            
            if args.format == "json":
                print(json.dumps(procedure_stats, indent=2))
            else:
                print("\nPer-Procedure Statistics:\n")
                for proc in procedure_stats:
                    print(f"Procedure ID: {proc['procedure_id']}")
                    print(f"  Average rating: {proc['average_rating'] or 'N/A'}")
                    print(f"  Rating count: {proc['rating_count']}")
                    print(f"  Success: {proc['success_count']}, Partial: {proc['partial_count']}, Failure: {proc['failure_count']}")
                    print(f"  Combined score: {proc['combined_score'] or 'N/A'}")
                    print(f"  Total feedback: {proc['total_feedback']}")
                    print()
        
        elif args.trends:
            # Time-based trends
            trends = analyzer.get_trends(granularity=args.granularity)
            
            if args.format == "json":
                print(json.dumps(trends, indent=2))
            else:
                print(f"\nFeedback Trends (granularity: {args.granularity}):\n")
                for period, data in sorted(trends.items()):
                    print(f"Period: {period}")
                    print(f"  Sessions: {data['session_count']}")
                    print(f"  Rated sessions: {data['rated_sessions']}")
                    print(f"  Average rating: {data['average_rating'] or 'N/A'}")
                    print(f"  Success: {data['success_count']}, Partial: {data['partial_count']}, Failure: {data['failure_count']}")
                    print(f"  Feedback entries: {data['feedback_count']}")
                    print()
        
        else:
            # Overall statistics
            stats = analyzer.get_statistics()
            print(_format_statistics(stats, args.format))
        
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return 1


def cmd_export(args: argparse.Namespace, collector: FeedbackCollector, analyzer: FeedbackAnalyzer) -> int:
    """Handle export subcommand."""
    try:
        output_path = Path(args.output)
        
        if args.format == "json":
            _export_to_json(
                collector=collector,
                analyzer=analyzer,
                output_path=output_path,
                sessions_only=args.sessions_only,
                feedback_only=args.feedback_only,
                procedure_id=args.procedure_id,
                since=args.since
            )
        elif args.format == "csv":
            _export_to_csv(
                collector=collector,
                analyzer=analyzer,
                output_path=output_path,
                sessions_only=args.sessions_only,
                feedback_only=args.feedback_only,
                procedure_id=args.procedure_id,
                since=args.since
            )
        else:
            logger.error(f"Unknown format: {args.format}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return 1


def cmd_list(args: argparse.Namespace, collector: FeedbackCollector) -> int:
    """Handle list subcommand."""
    try:
        if args.sessions:
            _list_sessions(
                collector=collector,
                session_id=args.session_id,
                limit=args.limit
            )
        elif args.feedback:
            _list_feedback(
                collector=collector,
                session_id=args.session_id,
                procedure_id=args.procedure_id,
                limit=args.limit
            )
        else:
            logger.error("Must specify --sessions or --feedback")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error listing data: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Utility script for collecting and managing feedback data"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to feedback database (default: from paths module)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommand
    add_parser = subparsers.add_parser('add', help='Add feedback')
    add_parser.add_argument('--session-id', type=str, help='Session ID (optional, generates new if not provided)')
    add_parser.add_argument('--fault-codes', type=str, help='Comma-separated fault codes (e.g., "P0301,P0302")')
    add_parser.add_argument('--rating', type=int, help='Rating from 1 to 5')
    add_parser.add_argument('--outcome', type=str, choices=['success', 'failure', 'partial'], help='Repair outcome')
    add_parser.add_argument('--procedure-id', type=str, help='Procedure ID for individual feedback')
    add_parser.add_argument('--feedback-text', type=str, help='Free-form feedback text')
    add_parser.add_argument('--selected-guide', type=str, help='ID of selected guide')
    add_parser.add_argument('--obd-data', type=str, help='JSON string or path to JSON file with OBD data')
    
    # Stats subcommand
    stats_parser = subparsers.add_parser('stats', help='View statistics')
    stats_parser.add_argument('--procedures', action='store_true', help='Show per-procedure statistics')
    stats_parser.add_argument('--trends', action='store_true', help='Show time-based trends')
    stats_parser.add_argument('--granularity', type=str, choices=['day', 'week', 'month'], default='day',
                             help='Time granularity for trends (default: day)')
    stats_parser.add_argument('--format', type=str, choices=['table', 'json'], default='table',
                             help='Output format (default: table)')
    stats_parser.add_argument('--min-rating-threshold', type=float,
                             help='Minimum rating threshold for filtering procedures')
    
    # Export subcommand
    export_parser = subparsers.add_parser('export', help='Export feedback data')
    export_parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                              help='Export format (default: json)')
    export_parser.add_argument('--output', type=str, required=True, help='Output file path')
    export_parser.add_argument('--sessions-only', action='store_true', help='Export only sessions')
    export_parser.add_argument('--feedback-only', action='store_true', help='Export only feedback entries')
    export_parser.add_argument('--procedure-id', type=str, help='Filter by procedure ID')
    export_parser.add_argument('--since', type=str, help='Filter by date (ISO format)')
    
    # List subcommand
    list_parser = subparsers.add_parser('list', help='List sessions or feedback entries')
    list_parser.add_argument('--sessions', action='store_true', help='List all sessions')
    list_parser.add_argument('--feedback', action='store_true', help='List all feedback entries')
    list_parser.add_argument('--session-id', type=str, help='Filter by session ID')
    list_parser.add_argument('--procedure-id', type=str, help='Filter by procedure ID')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize collector and analyzer
    try:
        paths = get_paths()
        db_path = args.db_path or str(paths.feedback_db)
        
        collector = FeedbackCollector(db_path)
        analyzer = FeedbackAnalyzer(db_path)
    except Exception as e:
        logger.error(f"Failed to initialize feedback collector: {e}")
        return 1
    
    # Route to appropriate command handler
    if args.command == 'add':
        return cmd_add(args, collector)
    elif args.command == 'stats':
        return cmd_stats(args, collector, analyzer)
    elif args.command == 'export':
        return cmd_export(args, collector, analyzer)
    elif args.command == 'list':
        return cmd_list(args, collector)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

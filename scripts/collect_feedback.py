#!/usr/bin/env python3
"""
Utility script for collecting and analyzing feedback.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feedback.collector import FeedbackCollector
from feedback.analyzer import FeedbackAnalyzer
from paths import get_paths
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """Collect and analyze feedback"""
    paths = get_paths()
    
    collector = FeedbackCollector(str(paths.feedback_db))
    analyzer = FeedbackAnalyzer(str(paths.feedback_db))
    
    stats = analyzer.get_statistics()
    
    print("Feedback Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Rated sessions: {stats['rated_sessions']}")
    print(f"  Average rating: {stats['average_rating']:.2f}")
    print(f"  Rating coverage: {stats['rating_coverage']:.1%}")
    print(f"  Repair outcomes: {stats['repair_outcomes']}")

if __name__ == "__main__":
    main()

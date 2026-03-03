#!/usr/bin/env python3
"""Thin wrapper - use 'mist build-kg' or 'python -m src.cli.main build-kg'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.commands.knowledge_graph import run


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build knowledge graph from BMW ISTA database")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--db-path", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    return run(
        incremental=args.incremental,
        output=args.output,
        db_path=args.db_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Thin wrapper - use 'mist migrate' or 'python -m src.cli.main migrate'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.commands.migrate import run

if __name__ == "__main__":
    sys.exit(run())

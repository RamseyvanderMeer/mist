#!/usr/bin/env python3
"""Thin wrapper - use 'mist readiness' or 'python -m src.cli.main readiness'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from src.commands.readiness import run

if __name__ == "__main__":
    sys.exit(run())

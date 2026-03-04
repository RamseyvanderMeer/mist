#!/usr/bin/env python3
"""
MIST CLI launcher. Run from project root to avoid conflict with npm 'mist':
  python mist.py fetch-bmwfault --limit 10
  python mist.py migrate
  python mist.py --help
"""
import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.cli.main import app

if __name__ == "__main__":
    app()

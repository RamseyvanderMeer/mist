#!/usr/bin/env python3
"""Launch chroma-mcp with env loaded from .env. Use this as MCP command to avoid hardcoding keys."""
import os
import sys
from pathlib import Path

# Load .env before chroma-mcp validates (dotenv-path in chroma-mcp runs too late)
_env_path = Path(os.environ.get("CHROMA_DOTENV_PATH", ""))
if not _env_path or not _env_path.exists():
    _env_path = Path.home() / ".cursor" / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# Pass through args to chroma-mcp (argv[0] = program name for its parser)
sys.argv = ["chroma-mcp"] + sys.argv[1:]
import chroma_mcp
chroma_mcp.main()

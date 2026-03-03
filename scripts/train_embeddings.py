#!/usr/bin/env python3
"""Thin wrapper - use 'mist train' or 'python -m src.cli.main train'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.commands.train import run


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Train embeddings from feedback data")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--embedding-config", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-source", type=str, default=None)
    args = p.parse_args()
    return run(
        config=args.config,
        resume=args.resume,
        embedding_config=args.embedding_config,
        log_level=args.log_level,
        device=args.device,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Thin wrapper - use 'mist scrape [forum|doc|example]' or 'python -m src.cli.main scrape'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from src.commands.scrape import run


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Run MIST web scraper")
    p.add_argument("--spider", choices=["forum", "doc", "example"], default="forum")
    p.add_argument("--output-dir", type=Path, default=Path("data/training/raw_data"))
    p.add_argument("--limit-items", type=int, default=None)
    p.add_argument("--limit-pages", type=int, default=None)
    p.add_argument("--url", type=str, default=None)
    p.add_argument("--search", action="store_true")
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--search-codes", action="store_true")
    p.add_argument("--re-scrape", action="store_true")
    args = p.parse_args()
    return run(
        spider=args.spider,
        output_dir=args.output_dir,
        limit_items=args.limit_items,
        limit_pages=args.limit_pages,
        url=args.url,
        search=args.search,
        targeted=args.targeted,
        search_codes=args.search_codes,
        re_scrape=args.re_scrape,
    )


if __name__ == "__main__":
    sys.exit(main())

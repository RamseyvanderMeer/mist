"""
End-to-end tests for run_scraper.py.

Runs the scraper with minimal mocks - real network for example.com.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_SCRAPER = PROJECT_ROOT / "scripts" / "run_scraper.py"


def test_example_spider_completes_with_limit():
    """Example spider runs to completion with --limit-items 1 and produces valid JSONL."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRAPER),
            "--spider",
            "example",
            "--limit-items",
            "1",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    output_file = Path("/tmp/scraper_output.jsonl")
    assert output_file.exists(), "Expected output file at /tmp/scraper_output.jsonl"

    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1, "Expected at least one JSONL line"

    for line in lines:
        if line.strip():
            record = json.loads(line)
            assert "url" in record
            assert "example.com" in record.get("url", "")


def test_doc_spider_completes_with_limit(tmp_path):
    """Doc spider runs with --limit-items 1 and writes to output dir."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRAPER),
            "--spider",
            "doc",
            "--output-dir",
            str(tmp_path),
            "--limit-items",
            "1",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    doc_dir = tmp_path / "documentation"
    jsonl_files = list(doc_dir.glob("*.jsonl")) if doc_dir.exists() else []
    assert len(jsonl_files) >= 1, f"Expected JSONL in {doc_dir}"

    content = jsonl_files[0].read_text(encoding="utf-8").strip()
    if content:
        for line in content.split("\n"):
            if line.strip():
                record = json.loads(line)
                assert "fault_codes" in record
                assert "source_url" in record


def test_run_scraper_help_includes_limit_options():
    """run_scraper --help shows --limit-items and --limit-pages."""
    result = subprocess.run(
        [sys.executable, str(RUN_SCRAPER), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "--limit-items" in result.stdout
    assert "--limit-pages" in result.stdout


def test_run_scraper_invalid_spider_exits_nonzero():
    """run_scraper exits with non-zero when spider resolution fails."""
    code = f"""
import sys
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
try:
    from scripts.run_scraper import get_spider_class
    get_spider_class('nonexistent')
except ValueError:
    sys.exit(1)
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=5,
    )
    # get_spider_class raises ValueError for unknown spider
    assert result.returncode == 1

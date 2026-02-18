"""
Unit tests for scripts.process_scraped_data.

Tests ScrapedDataProcessor and contract with JsonlWriterPipeline output.
"""
import pytest

from scripts.process_scraped_data import ScrapedDataProcessor


def test_process_record_accepts_jsonl_writer_output():
    """Contract: record from JsonlWriterPipeline is accepted by process_record."""
    # JsonlWriterPipeline writes dict(item) with repair_guide = repair_summary
    record = {
        "fault_codes": ["P0300", "P0420"],
        "obd_data": {"engine_rpm": 800.0, "coolant_temp": 95.0},
        "vehicle_context": {"make": "BMW", "model": "335i", "year": 2015},
        "repair_guide": "Replaced spark plugs and ignition coils. Cleared codes. Fixed.",
        "repair_summary": "Replaced spark plugs and ignition coils. Cleared codes. Fixed.",
        "outcome": "success",
        "source_url": "https://example.com/post",
        "source_type": "forum",
        "timestamp": "2025-01-15T12:00:00Z",
    }
    processor = ScrapedDataProcessor(min_quality_score=0.5)
    result = processor.process_record(record)
    assert result is not None
    assert result["fault_codes"] == ["P0300", "P0420"]
    assert "repair_guide" in result
    assert result["quality_score"] >= 0.5


def test_process_record_accepts_repair_summary_fallback():
    """Records with only repair_summary (no repair_guide) are processed."""
    record = {
        "fault_codes": ["P0300"],
        "obd_data": {},
        "vehicle_context": {"make": "Toyota", "model": "Camry", "year": 2018},
        "repair_summary": "Replaced the oxygen sensor. That fixed the P0300 code.",
        "outcome": "success",
        "source_url": "https://example.com",
    }
    processor = ScrapedDataProcessor(min_quality_score=0.5)
    result = processor.process_record(record)
    assert result is not None
    assert "repair_guide" in result
    assert result["repair_guide"]  # Should have content from repair_summary


def test_process_record_accepts_cause_to_solution():
    """Cause-to-solution records (no fault codes, symptoms + repair) are processed."""
    record = {
        "record_type": "cause_to_solution",
        "fault_codes": [],
        "obd_data": {},
        "vehicle_context": {"make": "BMW", "model": "335i"},
        "repair_summary": "Replaced ignition coil on cylinder 3. That fixed the rough idle.",
        "symptoms": "Rough idle, check engine light flashing",
        "outcome": "success",
        "source_url": "https://example.com/post",
    }
    processor = ScrapedDataProcessor(min_quality_score=0.5)
    result = processor.process_record(record)
    assert result is not None
    assert result["record_type"] == "cause_to_solution"
    assert result["fault_codes"] == []
    assert result["symptoms"] == "Rough idle, check engine light flashing"
    assert "repair_summary" in result
    assert result["quality_score"] >= 0.5

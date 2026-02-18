"""
Unit tests for scrapers.pipelines.validation.

Tests validate_fault_code, normalize_obd_data, and calculate_quality_score.
"""
import pytest
from scrapy.exceptions import DropItem

from scrapers.items import MistScrapedItem
from scrapers.pipelines.validation import (
    validate_fault_code,
    normalize_obd_data,
    calculate_quality_score,
    MistValidationPipeline,
)


class TestValidateFaultCode:
    """Tests for validate_fault_code."""

    def test_valid_p_codes(self):
        assert validate_fault_code("P0300") is True
        assert validate_fault_code("P0420") is True
        assert validate_fault_code("p0171") is True  # normalized to upper

    def test_valid_bmw_style_codes(self):
        # Pattern [A-Z][0-9][A-Z][0-9][0-9]
        assert validate_fault_code("A2B87") is True
        assert validate_fault_code("B3C45") is True

    def test_valid_manufacturer_codes(self):
        assert validate_fault_code("B1234") is True
        assert validate_fault_code("C5678") is True

    def test_invalid_codes(self):
        assert validate_fault_code("") is False
        assert validate_fault_code("P30") is False  # too short
        assert validate_fault_code("P03000") is False  # too long
        assert validate_fault_code("1234") is False  # no letter prefix
        assert validate_fault_code("ABC") is False
        assert validate_fault_code("P0300X") is False  # invalid suffix

    def test_whitespace_normalized(self):
        assert validate_fault_code("  P0300  ") is True
        assert validate_fault_code("p0420") is True


class TestNormalizeObdData:
    """Tests for normalize_obd_data."""

    def test_empty_input(self):
        assert normalize_obd_data({}) == {}
        assert normalize_obd_data(None) == {}

    def test_in_range_known_params(self):
        data = {"engine_rpm": 800, "coolant_temp": 95}
        result = normalize_obd_data(data)
        assert result == {"engine_rpm": 800.0, "coolant_temp": 95.0}

    def test_out_of_range_dropped(self):
        data = {"engine_rpm": 99999, "coolant_temp": 95}  # rpm out of 0-8000
        result = normalize_obd_data(data)
        assert "engine_rpm" not in result
        assert result["coolant_temp"] == 95.0

    def test_unknown_params_stored(self):
        data = {"custom_param": 42.5}
        result = normalize_obd_data(data)
        assert result == {"custom_param": 42.5}

    def test_key_normalization(self):
        data = {"Engine RPM": 800, "Coolant-Temp": 90}
        result = normalize_obd_data(data)
        assert "engine_rpm" in result
        assert "coolant_temp" in result

    def test_invalid_values_skipped(self):
        data = {"engine_rpm": "not_a_number", "coolant_temp": 95}
        result = normalize_obd_data(data)
        assert "engine_rpm" not in result
        assert result["coolant_temp"] == 95.0

    def test_fuel_trim_negative(self):
        data = {"fuel_trim_bank1": -5.5, "fuel_trim_bank2": 10}
        result = normalize_obd_data(data)
        assert result["fuel_trim_bank1"] == -5.5
        assert result["fuel_trim_bank2"] == 10.0


class TestCalculateQualityScore:
    """Tests for calculate_quality_score."""

    def test_minimal_item_low_score(self):
        item = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={},
            vehicle_context={},
            repair_summary="",
            outcome="unknown",
        )
        score = calculate_quality_score(item)
        assert score == 0.3  # fault codes only

    def test_full_item_high_score(self):
        item = MistScrapedItem(
            fault_codes=["P0300", "P0420"],
            obd_data={"engine_rpm": 800, "coolant_temp": 95, "maf_sensor": 4.2},
            vehicle_context={"make": "BMW", "model": "335i", "year": 2015},
            repair_summary="Replaced spark plugs and coils. Cleared codes. Fixed.",
            outcome="success",
        )
        score = calculate_quality_score(item)
        assert score >= 0.9
        assert score <= 1.0

    def test_repair_summary_length(self):
        item = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={},
            vehicle_context={"make": "BMW", "model": "335i"},
            repair_summary="x" * 60,  # >= 50 chars gets 0.3
            outcome="unknown",
        )
        score = calculate_quality_score(item)
        assert score >= 0.6

    def test_outcome_contribution(self):
        item_success = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={},
            vehicle_context={},
            repair_summary="x" * 60,
            outcome="success",
        )
        item_unknown = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={},
            vehicle_context={},
            repair_summary="x" * 60,
            outcome="unknown",
        )
        assert calculate_quality_score(item_success) > calculate_quality_score(item_unknown)


class TestMistValidationPipeline:
    """Tests for MistValidationPipeline."""

    def test_passes_valid_item(self):
        pipeline = MistValidationPipeline(min_quality=0.5)
        item = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={"engine_rpm": 800},
            vehicle_context={"make": "BMW"},
            repair_summary="Fixed by replacing sensor. " * 3,
            outcome="success",
        )
        result = pipeline.process_item(item, None)
        assert result is not None
        assert "confidence_score" in result
        assert result["fault_codes"] == ["P0300"]

    def test_drops_no_valid_fault_codes(self):
        pipeline = MistValidationPipeline(min_quality=0.3)
        item = MistScrapedItem(
            fault_codes=["INVALID", "BAD"],
            obd_data={},
            vehicle_context={},
            repair_summary="",
            outcome="unknown",
        )
        with pytest.raises(DropItem, match="No valid fault codes"):
            pipeline.process_item(item, None)

    def test_accepts_cause_to_solution_with_substantial_repair(self):
        pipeline = MistValidationPipeline(min_quality=0.3)
        item = MistScrapedItem(
            fault_codes=[],
            obd_data={},
            vehicle_context={"make": "BMW", "model": "335i"},
            repair_summary="Replaced ignition coil on cylinder 3. That fixed the rough idle.",
            symptoms="Rough idle, check engine light flashing",
            outcome="success",
        )
        result = pipeline.process_item(item, None)
        assert result is not None
        assert result["record_type"] == "cause_to_solution"
        assert result["fault_codes"] == []
        assert result["symptoms"] == "Rough idle, check engine light flashing"

    def test_drops_low_quality(self):
        pipeline = MistValidationPipeline(min_quality=0.9)
        item = MistScrapedItem(
            fault_codes=["P0300"],
            obd_data={},
            vehicle_context={},
            repair_summary="short",
            outcome="unknown",
        )
        with pytest.raises(DropItem, match="Quality score"):
            pipeline.process_item(item, None)

    def test_passes_through_non_mist_item(self):
        pipeline = MistValidationPipeline()
        other_item = {"url": "http://x.com", "html": "<html>"}
        result = pipeline.process_item(other_item, None)
        assert result == other_item

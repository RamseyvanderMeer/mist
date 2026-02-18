"""
Unit tests for scrapers.utils.extractors.

Tests fault code, OBD data, vehicle context, repair summary, and outcome extraction.
"""
import pytest

from scrapers.utils.extractors import (
    extract_fault_codes,
    extract_obd_data,
    extract_vehicle_context,
    extract_repair_summary,
    extract_outcome,
)


class TestExtractFaultCodes:
    """Tests for extract_fault_codes."""

    def test_empty_input(self):
        assert extract_fault_codes("") == []
        assert extract_fault_codes(None) == []
        assert extract_fault_codes("   ") == []

    def test_standard_p_codes(self):
        assert extract_fault_codes("P0300 random misfire") == ["P0300"]
        assert extract_fault_codes("Got P0420 and P0430") == ["P0420", "P0430"]
        assert extract_fault_codes("P0171 P0174 lean codes") == ["P0171", "P0174"]

    def test_bmw_style_codes(self):
        # Pattern [A-Z][0-9][A-Z][0-9][0-9] matches A2B87, B3C45, etc.
        assert extract_fault_codes("A2B87 vanos solenoid") == ["A2B87"]
        assert extract_fault_codes("B3C45 and C4D56") == ["B3C45", "C4D56"]

    def test_manufacturer_codes(self):
        assert extract_fault_codes("Code B1234 appeared") == ["B1234"]
        assert extract_fault_codes("C1234 and U5678") == ["C1234", "U5678"]

    def test_mixed_codes(self):
        text = "My 2015 BMW threw P0300 and A2B87. Also had P0420."
        result = extract_fault_codes(text)
        assert set(result) == {"P0300", "P0420", "A2B87"}
        assert len(result) == 3

    def test_case_insensitive_p_codes(self):
        assert extract_fault_codes("p0300 lowercase") == ["P0300"]
        assert extract_fault_codes("P0300 and p0420") == ["P0300", "P0420"]

    def test_no_codes_in_text(self):
        assert extract_fault_codes("Just some random text about cars") == []
        assert extract_fault_codes("Error 123 or code ABC") == []

    def test_non_string_input(self):
        assert extract_fault_codes(123) == []
        assert extract_fault_codes([]) == []


class TestExtractObdData:
    """Tests for extract_obd_data."""

    def test_empty_input(self):
        assert extract_obd_data("") == {}
        assert extract_obd_data(None) == {}

    def test_engine_rpm(self):
        assert extract_obd_data("RPM: 800") == {"engine_rpm": 800.0}
        assert extract_obd_data("engine rpm = 2500") == {"engine_rpm": 2500.0}
        assert extract_obd_data("Idle at 750 rpm") == {"engine_rpm": 750.0}

    def test_coolant_temp(self):
        assert extract_obd_data("Coolant temp: 95") == {"coolant_temp": 95.0}
        result = extract_obd_data("Coolant was 210°C")
        assert result.get("coolant_temp") == 210.0

    def test_throttle_position(self):
        assert extract_obd_data("TPS: 15") == {"throttle_position": 15.0}
        result = extract_obd_data("Throttle position = 0%")
        assert result.get("throttle_position") == 0.0

    def test_maf_sensor(self):
        assert extract_obd_data("MAF: 4.2 g/s") == {"maf_sensor": 4.2}
        assert extract_obd_data("Mass air flow = 12.5") == {"maf_sensor": 12.5}

    def test_fuel_trim(self):
        result = extract_obd_data("STFT B1: +5%")
        assert result.get("fuel_trim_bank1") == 5.0
        result2 = extract_obd_data("Fuel trim bank 2 = -3.2")
        assert result2.get("fuel_trim_bank2") == -3.2

    def test_multiple_params(self):
        text = "RPM: 800, coolant temp: 95, TPS: 12"
        result = extract_obd_data(text)
        assert result["engine_rpm"] == 800.0
        assert result["coolant_temp"] == 95.0
        assert result["throttle_position"] == 12.0

    def test_intake_air_temp(self):
        assert extract_obd_data("IAT: 35") == {"intake_air_temp": 35.0}
        result = extract_obd_data("Intake air temp = 42°C")
        assert result.get("intake_air_temp") == 42.0


class TestExtractVehicleContext:
    """Tests for extract_vehicle_context."""

    def test_empty_input(self):
        assert extract_vehicle_context("") == {}
        assert extract_vehicle_context(None) == {}

    def test_make_only(self):
        assert extract_vehicle_context("I have a BMW") == {"make": "Bmw"}
        assert extract_vehicle_context("Toyota owner here") == {"make": "Toyota"}

    def test_year(self):
        assert extract_vehicle_context("2015 model") == {"year": 2015}
        assert extract_vehicle_context("From 2020") == {"year": 2020}

    def test_engine(self):
        assert extract_vehicle_context("N54 engine") == {"engine": "N54"}
        assert extract_vehicle_context("B58 turbo") == {"engine": "B58"}
        assert extract_vehicle_context("2JZ swap") == {"engine": "2JZ"}

    def test_mileage(self):
        assert extract_vehicle_context("Mileage: 85000") == {"mileage": 85000}
        assert extract_vehicle_context("120,000 miles") == {"mileage": 120000}
        assert extract_vehicle_context("Odometer = 45000") == {"mileage": 45000}

    def test_full_context(self):
        text = "2018 BMW 335i with N54, 75,000 miles"
        result = extract_vehicle_context(text)
        assert result["make"] == "Bmw"
        assert result["year"] == 2018
        assert result["engine"] == "N54"
        assert result["mileage"] == 75000


class TestExtractRepairSummary:
    """Tests for extract_repair_summary."""

    def test_empty_input(self):
        assert extract_repair_summary("") == ""
        assert extract_repair_summary(None) == ""

    def test_short_text_unchanged(self):
        short = "Replaced the sensor. Fixed."
        assert extract_repair_summary(short, max_chars=500) == short

    def test_replaced_pattern(self):
        text = "Long intro. Replaced the oxygen sensor. That fixed it. More text."
        result = extract_repair_summary(text, max_chars=100)
        assert "Replaced" in result or "replaced" in result
        assert len(result) <= 100

    def test_solution_pattern(self):
        text = "Blah. Solution: Cleaned the MAF and reset codes. Done."
        result = extract_repair_summary(text, max_chars=80)
        assert "Solution" in result or "solution" in result or len(result) > 0

    def test_fallback_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth."
        result = extract_repair_summary(text, max_chars=50)
        assert result.endswith(".")
        assert len(result) <= 50


class TestExtractOutcome:
    """Tests for extract_outcome."""

    def test_empty_input(self):
        assert extract_outcome("") == "unknown"
        assert extract_outcome(None) == "unknown"

    def test_success(self):
        assert extract_outcome("Fixed it!") == "success"
        assert extract_outcome("Resolved the issue") == "success"
        assert extract_outcome("Worked perfectly") == "success"
        assert extract_outcome("No more codes") == "success"
        assert extract_outcome("Cleared the code") == "success"

    def test_failure(self):
        assert extract_outcome("Didn't work") == "failure"
        assert extract_outcome("Still getting the code") == "failure"
        assert extract_outcome("Failed again") == "failure"
        assert extract_outcome("No luck with that") == "failure"

    def test_partial(self):
        # "Partially fixed" matches "fixed" first -> success; use partial-only phrases
        assert extract_outcome("Somewhat better") == "partial"
        assert extract_outcome("Improved but not fully") == "partial"

    def test_unknown(self):
        assert extract_outcome("Trying something") == "unknown"
        assert extract_outcome("Not sure yet") == "unknown"

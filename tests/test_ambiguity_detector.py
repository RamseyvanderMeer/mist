"""
Tests for ambiguity detector module.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.retrieval.ambiguity_detector import (
    AmbiguityDetector,
    AmbiguityDetectorError,
    AmbiguityDetectorConfigurationError
)
from src.paths import Paths


class TestAmbiguityDetectorExceptions:
    """Test exception hierarchy."""
    
    def test_ambiguity_detector_error(self):
        """Test base exception."""
        with pytest.raises(AmbiguityDetectorError):
            raise AmbiguityDetectorError("Test error")
    
    def test_ambiguity_detector_configuration_error(self):
        """Test configuration error exception."""
        with pytest.raises(AmbiguityDetectorConfigurationError):
            raise AmbiguityDetectorConfigurationError("Config error")
        # Should also be instance of base exception
        assert issubclass(AmbiguityDetectorConfigurationError, AmbiguityDetectorError)


class TestAmbiguityDetectorInitialization:
    """Test AmbiguityDetector initialization."""
    
    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config = {
            "clarification": {
                "ambiguity_threshold": 0.7,
                "score_variance_threshold": 0.03,
                "critical_obd_params": ["engine_rpm", "coolant_temp", "vehicle_speed"]
            }
        }
        detector = AmbiguityDetector(config)
        assert detector.ambiguity_threshold == 0.7
        assert detector.score_variance_threshold == 0.03
        assert detector.critical_obd_params == ["engine_rpm", "coolant_temp", "vehicle_speed"]
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        assert detector.ambiguity_threshold == 0.65
        assert detector.score_variance_threshold == 0.02
        assert detector.critical_obd_params == ["engine_rpm", "coolant_temp"]
    
    def test_init_from_config_file(self):
        """Test initialization from config file."""
        paths = Paths()
        config_path = paths.retrieval_config
        detector = AmbiguityDetector(config_path)
        assert detector.ambiguity_threshold == 0.65
        assert detector.score_variance_threshold == 0.02
    
    def test_init_with_none(self):
        """Test initialization with None (uses default config file)."""
        detector = AmbiguityDetector(None)
        assert detector.ambiguity_threshold == 0.65
        assert detector.score_variance_threshold == 0.02
    
    def test_init_invalid_config_path(self):
        """Test initialization with invalid config path."""
        with pytest.raises(AmbiguityDetectorConfigurationError):
            AmbiguityDetector("/nonexistent/path/config.yaml")
    
    def test_init_invalid_config_type(self):
        """Test initialization with invalid config type."""
        with pytest.raises(AmbiguityDetectorConfigurationError):
            AmbiguityDetector(123)  # Invalid type


class TestTopScoreCheck:
    """Test top score threshold check."""
    
    def test_top_score_below_threshold(self):
        """Test detection when top score is below threshold."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.5, "procedure_id": "proc1"},
            {"combined_score": 0.4, "procedure_id": "proc2"}
        ]
        
        is_ambiguous, reason = detector._check_top_score(ranked_results)
        assert is_ambiguous is True
        assert "0.500" in reason
        assert "0.650" in reason
    
    def test_top_score_above_threshold(self):
        """Test no detection when top score is above threshold."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.8, "procedure_id": "proc1"},
            {"combined_score": 0.7, "procedure_id": "proc2"}
        ]
        
        is_ambiguous, reason = detector._check_top_score(ranked_results)
        assert is_ambiguous is False
        assert reason == ""
    
    def test_top_score_at_threshold(self):
        """Test no detection when top score equals threshold."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.65, "procedure_id": "proc1"}
        ]
        
        is_ambiguous, reason = detector._check_top_score(ranked_results)
        assert is_ambiguous is False
    
    def test_top_score_missing_combined_score(self):
        """Test handling when combined_score is missing."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"procedure_id": "proc1"}  # Missing combined_score
        ]
        
        is_ambiguous, reason = detector._check_top_score(ranked_results)
        assert is_ambiguous is True
        assert "0.000" in reason
    
    def test_top_score_empty_results(self):
        """Test handling of empty results."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        is_ambiguous, reason = detector._check_top_score([])
        assert is_ambiguous is True
        assert "No results" in reason


class TestScoreVarianceCheck:
    """Test score variance check."""
    
    def test_variance_below_threshold(self):
        """Test detection when variance is below threshold."""
        config = {"clarification": {"score_variance_threshold": 0.02}}
        detector = AmbiguityDetector(config)
        
        # Scores are very similar (low variance)
        ranked_results = [
            {"combined_score": 0.7, "procedure_id": "proc1"},
            {"combined_score": 0.71, "procedure_id": "proc2"},
            {"combined_score": 0.69, "procedure_id": "proc3"}
        ]
        
        is_ambiguous, reason = detector._check_score_variance(ranked_results)
        assert is_ambiguous is True
        assert "variance" in reason.lower()
        assert "0.02" in reason
    
    def test_variance_above_threshold(self):
        """Test no detection when variance is above threshold."""
        config = {"clarification": {"score_variance_threshold": 0.02}}
        detector = AmbiguityDetector(config)
        
        # Scores are different (high variance)
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"},
            {"combined_score": 0.5, "procedure_id": "proc2"},
            {"combined_score": 0.3, "procedure_id": "proc3"}
        ]
        
        is_ambiguous, reason = detector._check_score_variance(ranked_results)
        assert is_ambiguous is False
        assert reason == ""
    
    def test_variance_single_result(self):
        """Test that single result skips variance check."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.7, "procedure_id": "proc1"}
        ]
        
        is_ambiguous, reason = detector._check_score_variance(ranked_results)
        assert is_ambiguous is False
        assert reason == ""
    
    def test_variance_identical_scores(self):
        """Test detection when all scores are identical (variance = 0)."""
        config = {"clarification": {"score_variance_threshold": 0.02}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.7, "procedure_id": "proc1"},
            {"combined_score": 0.7, "procedure_id": "proc2"},
            {"combined_score": 0.7, "procedure_id": "proc3"}
        ]
        
        is_ambiguous, reason = detector._check_score_variance(ranked_results)
        assert is_ambiguous is True
        assert "variance" in reason.lower()
    
    def test_variance_top_n_limit(self):
        """Test that only top-N scores are considered (N=min(3, len))."""
        config = {"clarification": {"score_variance_threshold": 0.02}}
        detector = AmbiguityDetector(config)
        
        # Top 3 have low variance, but 4th is different
        ranked_results = [
            {"combined_score": 0.7, "procedure_id": "proc1"},
            {"combined_score": 0.71, "procedure_id": "proc2"},
            {"combined_score": 0.69, "procedure_id": "proc3"},
            {"combined_score": 0.2, "procedure_id": "proc4"}  # Should be ignored
        ]
        
        is_ambiguous, reason = detector._check_score_variance(ranked_results)
        assert is_ambiguous is True
        # Should only mention top 3 scores
        assert "top 3" in reason.lower() or "3" in reason


class TestMissingOBDParamsCheck:
    """Test missing OBD parameters check."""
    
    def test_obd_data_none(self):
        """Test detection when OBD data is None."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        is_ambiguous, reason = detector._check_missing_obd_params(None)
        assert is_ambiguous is True
        assert "OBD data missing" in reason
        assert "engine_rpm" in reason
        assert "coolant_temp" in reason
    
    def test_obd_data_missing_all_params(self):
        """Test detection when all critical params are missing."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        obd_data = {"vehicle_speed": 60, "throttle_position": 30}
        
        is_ambiguous, reason = detector._check_missing_obd_params(obd_data)
        assert is_ambiguous is True
        assert "engine_rpm" in reason
        assert "coolant_temp" in reason
    
    def test_obd_data_missing_one_param(self):
        """Test detection when one critical param is missing."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        obd_data = {"engine_rpm": 2000}  # Missing coolant_temp
        
        is_ambiguous, reason = detector._check_missing_obd_params(obd_data)
        assert is_ambiguous is True
        assert "coolant_temp" in reason
        assert "engine_rpm" not in reason or "Missing" in reason
    
    def test_obd_data_all_params_present(self):
        """Test no detection when all critical params are present."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90,
            "vehicle_speed": 60
        }
        
        is_ambiguous, reason = detector._check_missing_obd_params(obd_data)
        assert is_ambiguous is False
        assert reason == ""
    
    def test_obd_data_custom_critical_params(self):
        """Test with custom critical OBD parameters."""
        config = {
            "clarification": {
                "critical_obd_params": ["engine_rpm", "vehicle_speed"]
            }
        }
        detector = AmbiguityDetector(config)
        
        obd_data = {"engine_rpm": 2000}  # Missing vehicle_speed
        
        is_ambiguous, reason = detector._check_missing_obd_params(obd_data)
        assert is_ambiguous is True
        assert "vehicle_speed" in reason
        assert "coolant_temp" not in reason


class TestDetectMethod:
    """Test main detect method combining all checks."""
    
    def test_detect_not_ambiguous(self):
        """Test detection when all checks pass."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"},
            {"combined_score": 0.7, "procedure_id": "proc2"},
            {"combined_score": 0.5, "procedure_id": "proc3"}
        ]
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90
        }
        
        is_ambiguous, reason = detector.detect(ranked_results, obd_data)
        assert is_ambiguous is False
        assert reason == ""
    
    def test_detect_ambiguous_top_score(self):
        """Test detection when top score is low."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.5, "procedure_id": "proc1"}
        ]
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90
        }
        
        is_ambiguous, reason = detector.detect(ranked_results, obd_data)
        assert is_ambiguous is True
        assert "Top score" in reason or "below threshold" in reason
    
    def test_detect_ambiguous_variance(self):
        """Test detection when variance is low."""
        config = {
            "clarification": {
                "ambiguity_threshold": 0.65,
                "score_variance_threshold": 0.02
            }
        }
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.7, "procedure_id": "proc1"},
            {"combined_score": 0.71, "procedure_id": "proc2"},
            {"combined_score": 0.69, "procedure_id": "proc3"}
        ]
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90
        }
        
        is_ambiguous, reason = detector.detect(ranked_results, obd_data)
        assert is_ambiguous is True
        assert "variance" in reason.lower()
    
    def test_detect_ambiguous_missing_obd(self):
        """Test detection when OBD data is missing."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, None)
        assert is_ambiguous is True
        assert "OBD" in reason or "missing" in reason.lower()
    
    def test_detect_multiple_reasons(self):
        """Test detection with multiple ambiguous conditions."""
        config = {
            "clarification": {
                "ambiguity_threshold": 0.65,
                "score_variance_threshold": 0.02
            }
        }
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.5, "procedure_id": "proc1"},
            {"combined_score": 0.51, "procedure_id": "proc2"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, None)
        assert is_ambiguous is True
        # Should contain multiple reasons separated by semicolon
        assert ";" in reason or len(reason) > 50
    
    def test_detect_empty_results(self):
        """Test detection with empty results."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        is_ambiguous, reason = detector.detect([], None)
        assert is_ambiguous is True
        assert "No results" in reason
    
    def test_detect_single_result_skips_variance(self):
        """Test that single result skips variance check."""
        config = {
            "clarification": {
                "ambiguity_threshold": 0.65,
                "score_variance_threshold": 0.02
            }
        }
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"}
        ]
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90
        }
        
        is_ambiguous, reason = detector.detect(ranked_results, obd_data)
        # Should not be ambiguous (top score high, OBD present, variance skipped)
        assert is_ambiguous is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_detect_with_missing_combined_score(self):
        """Test detection when combined_score is missing from results."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"procedure_id": "proc1"},  # Missing combined_score
            {"procedure_id": "proc2"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, None)
        assert is_ambiguous is True
    
    def test_detect_with_zero_scores(self):
        """Test detection with zero scores."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.0, "procedure_id": "proc1"},
            {"combined_score": 0.0, "procedure_id": "proc2"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, None)
        assert is_ambiguous is True
    
    def test_detect_with_negative_scores(self):
        """Test detection with negative scores (should be clamped)."""
        config = {"clarification": {"ambiguity_threshold": 0.65}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": -0.1, "procedure_id": "proc1"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, None)
        assert is_ambiguous is True
    
    def test_detect_with_empty_obd_dict(self):
        """Test detection with empty OBD dict."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"}
        ]
        
        is_ambiguous, reason = detector.detect(ranked_results, {})
        assert is_ambiguous is True
        assert "Missing" in reason or "OBD" in reason
    
    def test_detect_with_extra_obd_params(self):
        """Test that extra OBD params don't cause issues."""
        config = {"clarification": {}}
        detector = AmbiguityDetector(config)
        
        ranked_results = [
            {"combined_score": 0.9, "procedure_id": "proc1"}
        ]
        obd_data = {
            "engine_rpm": 2000,
            "coolant_temp": 90,
            "vehicle_speed": 60,
            "throttle_position": 30,
            "extra_param": "value"
        }
        
        is_ambiguous, reason = detector.detect(ranked_results, obd_data)
        assert is_ambiguous is False

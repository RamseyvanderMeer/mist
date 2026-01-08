"""
Unit tests for prompt template manager.

Tests cover:
- Loading templates from YAML config
- Variable substitution with real data examples
- Error handling for missing config, templates, and variables
- Edge cases and validation
"""
import tempfile
import yaml
from pathlib import Path
import pytest

from src.llm.prompt_templates import PromptTemplates
from src.paths import Paths


class TestPromptTemplatesInitialization:
    """Test PromptTemplates initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default config path."""
        templates = PromptTemplates()
        assert templates.config_path.exists()
        assert templates.config_path.name == "llm_config.yaml"
        assert "clarification" in templates._templates
        assert "query_expansion" in templates._templates
    
    def test_init_with_custom_config_path(self):
        """Test initialization with custom config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_llm_config.yaml"
            
            # Create minimal valid config
            config = {
                "prompts": {
                    "clarification": {
                        "system": "Test system prompt",
                        "user_template": "Fault Codes: {fault_codes}\nOBD Data: {obd_data}\nTop Recommendations: {top_candidates}"
                    },
                    "query_expansion": {
                        "system": "Test expansion system",
                        "user_template": "Original Query: {original_query}\nUser Responses: {user_responses}"
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            templates = PromptTemplates(config_path=config_path)
            assert templates.config_path == config_path
            assert templates._templates["clarification"]["system"] == "Test system prompt"
    
    def test_init_missing_config_file(self):
        """Test initialization with missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"
            
            with pytest.raises(FileNotFoundError) as exc_info:
                PromptTemplates(config_path=config_path)
            
            assert "not found" in str(exc_info.value).lower()
            assert str(config_path) in str(exc_info.value)
    
    def test_init_missing_prompts_section(self):
        """Test initialization with config missing prompts section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_config.yaml"
            
            config = {
                "providers": {
                    "primary": "openai"
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            with pytest.raises(KeyError) as exc_info:
                PromptTemplates(config_path=config_path)
            
            assert "prompts" in str(exc_info.value).lower()
    
    def test_init_missing_required_template(self):
        """Test initialization with missing required template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "incomplete_config.yaml"
            
            config = {
                "prompts": {
                    "clarification": {
                        "system": "Test",
                        "user_template": "Test {var}"
                    }
                    # Missing query_expansion
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            with pytest.raises(KeyError) as exc_info:
                PromptTemplates(config_path=config_path)
            
            assert "query_expansion" in str(exc_info.value).lower()


class TestClarificationPrompt:
    """Test clarification prompt generation with real data examples."""
    
    @pytest.fixture
    def templates(self):
        """Create PromptTemplates instance."""
        return PromptTemplates()
    
    def test_clarification_prompt_basic(self, templates):
        """Test basic clarification prompt with simple data."""
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        top_candidates = "Engine misfire detected"
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        assert "system" in prompt
        assert "user" in prompt
        assert isinstance(prompt["system"], str)
        assert isinstance(prompt["user"], str)
        assert len(prompt["system"]) > 0
        assert len(prompt["user"]) > 0
    
    def test_clarification_prompt_real_fault_codes(self, templates):
        """Test with real OBD-II fault codes."""
        # Real BMW fault codes from diagnostic systems
        fault_codes = [
            "P0301",  # Cylinder 1 Misfire Detected
            "P0302",  # Cylinder 2 Misfire Detected
            "P0171",  # System Too Lean (Bank 1)
        ]
        
        # Real OBD-II sensor data
        obd_data = {
            "rpm": 750,
            "coolant_temp": 90,
            "intake_temp": 25,
            "maf": 12.5,
            "throttle_position": 15.2,
            "fuel_pressure": 45.0,
            "o2_sensor_1": 0.45,
            "timing_advance": 12.5
        }
        
        top_candidates = (
            "1. Ignition coil replacement for cylinders 1 and 2\n"
            "2. Spark plug inspection and replacement\n"
            "3. Fuel injector cleaning for bank 1"
        )
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        # Verify all fault codes appear in user prompt
        assert "P0301" in prompt["user"]
        assert "P0302" in prompt["user"]
        assert "P0171" in prompt["user"]
        
        # Verify OBD data appears (as string representation)
        assert "750" in prompt["user"] or "rpm" in prompt["user"].lower()
        
        # Verify top candidates appear
        assert "Ignition coil" in prompt["user"] or "misfire" in prompt["user"].lower()
    
    def test_clarification_prompt_multiple_fault_codes(self, templates):
        """Test with multiple fault codes."""
        fault_codes = [
            "P0420",  # Catalyst System Efficiency Below Threshold
            "P0430",  # Catalyst System Efficiency Below Threshold (Bank 2)
            "P0174",  # System Too Lean (Bank 2)
            "P0171",  # System Too Lean (Bank 1)
        ]
        
        obd_data = {
            "rpm": 2000,
            "speed": 65,
            "coolant_temp": 95,
            "catalyst_temp_bank1": 650,
            "catalyst_temp_bank2": 620,
            "o2_sensor_1": 0.12,
            "o2_sensor_2": 0.15,
            "fuel_trim_bank1": 15.5,
            "fuel_trim_bank2": 18.2
        }
        
        top_candidates = (
            "Catalyst replacement recommended for both banks. "
            "Check oxygen sensors and fuel trim values."
        )
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        # Verify all codes are included
        for code in fault_codes:
            assert code in prompt["user"]
        
        # Verify system prompt is present
        assert "diagnostic assistant" in prompt["system"].lower()
    
    def test_clarification_prompt_empty_fault_codes(self, templates):
        """Test with empty fault codes list."""
        fault_codes = []
        obd_data = {"rpm": 0}
        top_candidates = "No fault codes detected"
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        # Should still work, just with empty string for fault codes
        assert "system" in prompt
        assert "user" in prompt
    
    def test_clarification_prompt_complex_obd_data(self, templates):
        """Test with complex OBD data structure."""
        fault_codes = ["P0128"]  # Coolant Thermostat (Coolant Temperature Below Thermostat Regulating Temperature)
        
        # Comprehensive OBD-II data
        obd_data = {
            "engine_rpm": 850,
            "vehicle_speed": 0,
            "throttle_position": 0.0,
            "engine_coolant_temp": 65,  # Below normal operating temp
            "intake_air_temp": 22,
            "maf_air_flow": 3.2,
            "fuel_pressure": 48.5,
            "intake_manifold_pressure": 98.5,
            "timing_advance": 8.5,
            "fuel_level": 75,
            "barometric_pressure": 101.3,
            "o2_sensor_1_voltage": 0.45,
            "o2_sensor_2_voltage": 0.42,
            "catalyst_temp_bank1": 420,
            "short_term_fuel_trim_bank1": 2.5,
            "long_term_fuel_trim_bank1": 0.0,
            "short_term_fuel_trim_bank2": 3.1,
            "long_term_fuel_trim_bank2": 0.5
        }
        
        top_candidates = (
            "1. Replace engine coolant thermostat\n"
            "2. Check coolant temperature sensor\n"
            "3. Verify proper engine warm-up cycle"
        )
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        # Verify OBD data is included (will be string representation)
        assert "65" in prompt["user"] or "coolant" in prompt["user"].lower()
        assert "P0128" in prompt["user"]
    
    def test_clarification_prompt_missing_variable(self, templates):
        """Test that missing variables raise ValueError."""
        # Try to call with wrong signature to trigger validation
        # We'll use _substitute_template directly to test validation
        template = "Test {missing_var} template"
        variables = {"other_var": "value"}
        
        with pytest.raises(ValueError) as exc_info:
            templates._substitute_template(template, variables)
        
        assert "missing_var" in str(exc_info.value).lower()
        assert "missing" in str(exc_info.value).lower()


class TestQueryExpansionPrompt:
    """Test query expansion prompt generation with real data examples."""
    
    @pytest.fixture
    def templates(self):
        """Create PromptTemplates instance."""
        return PromptTemplates()
    
    def test_query_expansion_prompt_basic(self, templates):
        """Test basic query expansion prompt."""
        original_query = "Engine misfire"
        user_responses = ["Yes, happens at idle"]
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        assert "system" in prompt
        assert "user" in prompt
        assert isinstance(prompt["system"], str)
        assert isinstance(prompt["user"], str)
        assert len(prompt["system"]) > 0
        assert len(prompt["user"]) > 0
    
    def test_query_expansion_prompt_real_scenario(self, templates):
        """Test with real diagnostic scenario."""
        original_query = (
            "Multiple cylinder misfire detected with P0301, P0302, P0303 fault codes. "
            "Vehicle experiences rough idle and loss of power under acceleration."
        )
        
        user_responses = [
            "Yes, the misfire occurs primarily at idle speed",
            "The check engine light flashes during acceleration",
            "Vehicle has 85,000 miles and spark plugs were last changed at 60,000 miles",
            "No recent fuel system service performed"
        ]
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        # Verify original query appears
        assert "P0301" in prompt["user"]
        assert "misfire" in prompt["user"].lower()
        
        # Verify user responses appear
        assert "idle" in prompt["user"].lower()
        assert "85,000" in prompt["user"] or "miles" in prompt["user"].lower()
        
        # Verify system prompt
        assert "query expansion" in prompt["system"].lower() or "expansion" in prompt["system"].lower()
    
    def test_query_expansion_prompt_multiple_responses(self, templates):
        """Test with multiple user responses."""
        original_query = "Catalyst efficiency below threshold"
        
        user_responses = [
            "Vehicle has 120,000 miles",
            "Catalytic converter was never replaced",
            "Check engine light has been on for 3 months",
            "Fuel economy has decreased noticeably",
            "No unusual exhaust smells"
        ]
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        # Verify all responses are included
        assert "120,000" in prompt["user"] or "miles" in prompt["user"].lower()
        assert "Catalytic converter" in prompt["user"] or "catalyst" in prompt["user"].lower()
        assert "3 months" in prompt["user"] or "months" in prompt["user"].lower()
    
    def test_query_expansion_prompt_empty_responses(self, templates):
        """Test with empty user responses list."""
        original_query = "Engine overheating"
        user_responses = []
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        # Should still work, just with empty string for responses
        assert "system" in prompt
        assert "user" in prompt
        assert "overheating" in prompt["user"].lower()
    
    def test_query_expansion_prompt_complex_query(self, templates):
        """Test with complex original query."""
        original_query = (
            "BMW 328i 2015 - Fault codes P0171, P0174 indicate lean condition on both banks. "
            "OBD data shows: MAF reading 8.2 g/s at idle (normal is 12-15 g/s), "
            "fuel trim bank 1: +18%, bank 2: +22%. Vehicle has rough idle and hesitation. "
            "Air filter was replaced 2 months ago."
        )
        
        user_responses = [
            "Yes, the rough idle is more noticeable when the A/C is on",
            "I recently had a vacuum leak repaired on the intake manifold",
            "The fuel economy has been getting worse over the past month",
            "No fuel smell detected"
        ]
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        # Verify complex query details are preserved
        assert "P0171" in prompt["user"]
        assert "P0174" in prompt["user"]
        assert "MAF" in prompt["user"] or "maf" in prompt["user"].lower()
        
        # Verify user responses are included
        assert "A/C" in prompt["user"] or "vacuum" in prompt["user"].lower()
    
    def test_query_expansion_prompt_missing_variable(self, templates):
        """Test that missing variables raise ValueError."""
        template = "Query: {original_query}\nResponses: {missing_var}"
        variables = {"original_query": "test"}
        
        with pytest.raises(ValueError) as exc_info:
            templates._substitute_template(template, variables)
        
        assert "missing_var" in str(exc_info.value).lower()


class TestVariableExtraction:
    """Test variable extraction from templates."""
    
    @pytest.fixture
    def templates(self):
        """Create PromptTemplates instance."""
        return PromptTemplates()
    
    def test_extract_single_variable(self, templates):
        """Test extracting single variable."""
        template = "Hello {name}"
        variables = templates._extract_variables(template)
        assert variables == ["name"]
    
    def test_extract_multiple_variables(self, templates):
        """Test extracting multiple variables."""
        template = "Fault Codes: {fault_codes}\nOBD Data: {obd_data}\nTop Recommendations: {top_candidates}"
        variables = templates._extract_variables(template)
        assert len(variables) == 3
        assert "fault_codes" in variables
        assert "obd_data" in variables
        assert "top_candidates" in variables
    
    def test_extract_no_variables(self, templates):
        """Test template with no variables."""
        template = "This is a plain template with no variables"
        variables = templates._extract_variables(template)
        assert variables == []
    
    def test_extract_duplicate_variables(self, templates):
        """Test template with duplicate variable names."""
        template = "{var} and {var} again"
        variables = templates._extract_variables(template)
        # Should return unique variables
        assert variables == ["var"]
    
    def test_extract_nested_braces(self, templates):
        """Test template with nested braces (should not match)."""
        template = "Test {{escaped}} and {real_var}"
        variables = templates._extract_variables(template)
        # Should only extract real_var, not escaped
        assert "real_var" in variables
        assert "escaped" not in variables


class TestErrorHandling:
    """Test error handling for various edge cases."""
    
    def test_missing_template_section(self):
        """Test handling of missing template section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.yaml"
            
            config = {
                "prompts": {
                    "clarification": {
                        "system": "Test"
                        # Missing user_template
                    },
                    "query_expansion": {
                        "system": "Test",
                        "user_template": "Test"
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            with pytest.raises(KeyError) as exc_info:
                PromptTemplates(config_path=config_path)
            
            assert "user_template" in str(exc_info.value).lower() or "clarification" in str(exc_info.value).lower()
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            
            # Write invalid YAML
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")
            
            with pytest.raises((ValueError, yaml.YAMLError)):
                PromptTemplates(config_path=config_path)
    
    def test_empty_config_file(self):
        """Test handling of empty config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty.yaml"
            
            # Create empty file
            config_path.touch()
            
            with pytest.raises(ValueError) as exc_info:
                PromptTemplates(config_path=config_path)
            
            assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


class TestIntegrationWithRealConfig:
    """Integration tests with actual project config file."""
    
    def test_loads_actual_config(self):
        """Test that templates load from actual project config."""
        templates = PromptTemplates()
        
        # Verify templates are loaded
        assert "clarification" in templates._templates
        assert "query_expansion" in templates._templates
        
        # Verify structure
        clarification = templates._templates["clarification"]
        assert "system" in clarification
        assert "user_template" in clarification
        
        query_expansion = templates._templates["query_expansion"]
        assert "system" in query_expansion
        assert "user_template" in query_expansion
    
    def test_real_config_clarification_prompt(self):
        """Test clarification prompt with real config."""
        templates = PromptTemplates()
        
        fault_codes = ["P0301", "P0302"]
        obd_data = {
            "rpm": 750,
            "coolant_temp": 90,
            "maf": 12.5
        }
        top_candidates = "Ignition coil replacement recommended"
        
        prompt = templates.get_clarification_prompt(
            fault_codes=fault_codes,
            obd_data=obd_data,
            top_candidates=top_candidates
        )
        
        # Verify it uses the actual config template structure
        assert "diagnostic assistant" in prompt["system"].lower()
        assert "fault codes" in prompt["user"].lower() or "Fault Codes" in prompt["user"]
        assert "P0301" in prompt["user"]
        assert "P0302" in prompt["user"]
    
    def test_real_config_query_expansion_prompt(self):
        """Test query expansion prompt with real config."""
        templates = PromptTemplates()
        
        original_query = "Engine misfire at idle"
        user_responses = [
            "Yes, happens when A/C is on",
            "Vehicle has 90,000 miles"
        ]
        
        prompt = templates.get_query_expansion_prompt(
            original_query=original_query,
            user_responses=user_responses
        )
        
        # Verify it uses the actual config template structure
        assert "query expansion" in prompt["system"].lower() or "expansion" in prompt["system"].lower()
        assert "original query" in prompt["user"].lower() or "Original Query" in prompt["user"]
        assert "misfire" in prompt["user"].lower()
        assert "A/C" in prompt["user"] or "90,000" in prompt["user"]

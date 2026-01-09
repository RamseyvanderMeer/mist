"""
Unit tests for clarification question generator.

Tests cover:
- Initialization with default and custom configs
- Question generation with mocked LLM providers
- Question parsing (numbered, bulleted, plain text)
- Error handling (API errors, rate limits, configuration errors)
- Edge cases and validation
"""
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.retrieval.clarification_generator import (
    ClarificationGenerator,
    ClarificationGeneratorError,
    ClarificationGenerationError
)
from src.llm.provider import (
    LLMProvider,
    LLMAPIError,
    LLMRateLimitError,
    LLMConfigurationError
)
from src.paths import Paths


class TestClarificationGeneratorExceptions:
    """Test exception hierarchy."""
    
    def test_clarification_generator_error(self):
        """Test base exception."""
        with pytest.raises(ClarificationGeneratorError):
            raise ClarificationGeneratorError("Test error")
    
    def test_clarification_generation_error(self):
        """Test generation error exception."""
        with pytest.raises(ClarificationGenerationError):
            raise ClarificationGenerationError("Generation error")
        # Should also be instance of base exception
        assert issubclass(ClarificationGenerationError, ClarificationGeneratorError)


class TestClarificationGeneratorInitialization:
    """Test ClarificationGenerator initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default config path."""
        # Mock LLM provider to avoid needing real API keys
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            generator = ClarificationGenerator()
            assert generator.config_path.exists()
            assert generator.config_path.name == "llm_config.yaml"
            assert generator.prompt_templates is not None
            assert generator.llm_provider is not None
    
    def test_init_with_custom_config_path(self):
        """Test initialization with custom config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_llm_config.yaml"
            
            # Create minimal valid config
            config = {
                "providers": {
                    "primary": "openai",
                    "fallback": []
                },
                "openai": {
                    "model": "gpt-4o",
                    "api_key_env": "OPENAI_API_KEY",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
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
            
            # Mock LLM provider
            mock_provider = Mock(spec=LLMProvider)
            mock_provider.get_model_info.return_value = {
                "provider": "openai",
                "model_name": "gpt-4o"
            }
            
            with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
                generator = ClarificationGenerator(config_path=str(config_path))
                assert generator.config_path == config_path
                assert generator.llm_config == config
    
    def test_init_missing_config_file(self):
        """Test initialization with missing config file."""
        with pytest.raises(FileNotFoundError):
            ClarificationGenerator("/nonexistent/path/config.yaml")
    
    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            with pytest.raises(ValueError):
                ClarificationGenerator(str(config_path))
    
    def test_init_missing_providers_section(self):
        """Test initialization with missing providers section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config = {
                "prompts": {
                    "clarification": {
                        "system": "Test",
                        "user_template": "Test {fault_codes}"
                    },
                    "query_expansion": {
                        "system": "Test",
                        "user_template": "Test {original_query}"
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            with pytest.raises(ValueError) as exc_info:
                ClarificationGenerator(str(config_path))
            assert "providers" in str(exc_info.value).lower()
    
    def test_init_provider_failure(self):
        """Test initialization when all providers fail."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', side_effect=RuntimeError("All providers failed")):
            with pytest.raises(RuntimeError) as exc_info:
                ClarificationGenerator()
            assert "All LLM providers failed" in str(exc_info.value)


class TestQuestionGeneration:
    """Test question generation with mocked LLM providers."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        return provider
    
    @pytest.fixture
    def generator(self, mock_provider):
        """Create ClarificationGenerator with mocked provider."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            return ClarificationGenerator()
    
    def test_generate_questions_numbered_format(self, generator, mock_provider):
        """Test question generation with numbered format."""
        mock_provider.generate.return_value = (
            "1. What is the vehicle's mileage?\n"
            "2. When did the fault codes first appear?\n"
            "3. Are there any unusual sounds or symptoms?"
        )
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Ignition coil replacement", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) == 3
        assert "mileage" in questions[0].lower()
        assert "fault codes" in questions[1].lower() or "appear" in questions[1].lower()
        assert "sounds" in questions[2].lower() or "symptoms" in questions[2].lower()
    
    def test_generate_questions_bulleted_format(self, generator, mock_provider):
        """Test question generation with bulleted format."""
        mock_provider.generate.return_value = (
            "- What is the vehicle's mileage?\n"
            "- When did the fault codes first appear?\n"
            "- Are there any unusual sounds?"
        )
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) == 3
        assert all("?" in q or len(q) > 10 for q in questions)
    
    def test_generate_questions_plain_text(self, generator, mock_provider):
        """Test question generation with plain text format."""
        mock_provider.generate.return_value = (
            "What is the vehicle's mileage? "
            "When did the fault codes first appear? "
            "Are there any unusual sounds?"
        )
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) > 0
        assert len(questions) <= 3
    
    def test_generate_questions_single_question(self, generator, mock_provider):
        """Test question generation with single question."""
        mock_provider.generate.return_value = "What is the vehicle's mileage?"
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) == 1
        assert "mileage" in questions[0].lower()
    
    def test_generate_questions_max_three(self, generator, mock_provider):
        """Test that maximum 3 questions are returned."""
        mock_provider.generate.return_value = (
            "1. Question one?\n"
            "2. Question two?\n"
            "3. Question three?\n"
            "4. Question four?\n"
            "5. Question five?"
        )
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) <= 3
    
    def test_generate_questions_with_multiple_candidates(self, generator, mock_provider):
        """Test question generation with multiple candidates."""
        mock_provider.generate.return_value = "What is the vehicle's mileage?"
        
        fault_codes = ["P0301", "P0302"]
        obd_data = {"rpm": 2000, "coolant_temp": 90}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Ignition coil", "combined_score": 0.8},
            {"procedure_id": "proc2", "procedure_name": "Spark plug", "combined_score": 0.7},
            {"procedure_id": "proc3", "title": "Fuel injector", "combined_score": 0.6}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) == 1
        # Verify that prompt was formatted correctly (check mock call)
        call_args = mock_provider.generate.call_args
        assert call_args is not None
        # generate() is called with messages as keyword arg
        messages = call_args.kwargs.get('messages', [])
        assert len(messages) >= 2
        assert "P0301" in messages[1]["content"]
        assert "P0302" in messages[1]["content"]
    
    def test_generate_questions_empty_candidates(self, generator, mock_provider):
        """Test question generation with empty candidates."""
        mock_provider.generate.return_value = "What is the vehicle's mileage?"
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = []
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        assert len(questions) == 1
        # Verify that "No recommendations" appears in prompt
        call_args = mock_provider.generate.call_args
        assert call_args is not None
        # generate() is called with messages as keyword arg
        messages = call_args.kwargs.get('messages', [])
        assert len(messages) >= 2
        assert "No recommendations" in messages[1]["content"] or "available" in messages[1]["content"].lower()


class TestQuestionParsing:
    """Test question parsing from various LLM response formats."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        return provider
    
    @pytest.fixture
    def generator(self, mock_provider):
        """Create ClarificationGenerator with mocked provider."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            return ClarificationGenerator()
    
    def test_parse_numbered_questions(self, generator):
        """Test parsing numbered questions."""
        response = "1. What is the mileage?\n2. When did it start?\n3. Any sounds?"
        questions = generator._parse_questions(response)
        
        assert len(questions) == 3
        assert "mileage" in questions[0].lower()
        assert "start" in questions[1].lower() or "when" in questions[1].lower()
        assert "sounds" in questions[2].lower()
    
    def test_parse_bulleted_questions(self, generator):
        """Test parsing bulleted questions."""
        response = "- What is the mileage?\n- When did it start?\n- Any sounds?"
        questions = generator._parse_questions(response)
        
        assert len(questions) == 3
        assert all(len(q) > 5 for q in questions)
    
    def test_parse_mixed_format(self, generator):
        """Test parsing mixed format questions."""
        response = "1. What is the mileage?\n- When did it start?\n2. Any sounds?"
        questions = generator._parse_questions(response)
        
        assert len(questions) >= 2
    
    def test_parse_questions_with_parentheses(self, generator):
        """Test parsing numbered questions with parentheses."""
        response = "1) What is the mileage?\n2) When did it start?"
        questions = generator._parse_questions(response)
        
        assert len(questions) == 2
        assert "mileage" in questions[0].lower()
    
    def test_parse_questions_with_extra_text(self, generator):
        """Test parsing questions with extra explanatory text."""
        response = (
            "Here are some clarifying questions:\n"
            "1. What is the vehicle's mileage?\n"
            "2. When did the fault codes first appear?\n"
            "These questions will help narrow down the diagnosis."
        )
        questions = generator._parse_questions(response)
        
        # Should filter out "Here are some clarifying questions:" header
        assert len(questions) >= 2
        assert len(questions) <= 3
        # Check that numbered questions are parsed
        assert any("mileage" in q.lower() for q in questions)
        assert any("fault codes" in q.lower() or "appear" in q.lower() for q in questions)
    
    def test_parse_empty_response(self, generator):
        """Test parsing empty response."""
        questions = generator._parse_questions("")
        assert questions == []
    
    def test_parse_whitespace_only(self, generator):
        """Test parsing whitespace-only response."""
        questions = generator._parse_questions("   \n\n   ")
        assert questions == []
    
    def test_parse_questions_filters_instructions(self, generator):
        """Test that parsing filters out instruction-like text."""
        response = (
            "Focus on missing information.\n"
            "1. What is the mileage?\n"
            "Return only the questions.\n"
            "2. When did it start?"
        )
        questions = generator._parse_questions(response)
        
        # Should filter out "Focus on" and "Return only" lines
        assert len(questions) == 2
        assert all("focus" not in q.lower() and "return" not in q.lower() for q in questions)
    
    def test_parse_questions_limits_to_three(self, generator):
        """Test that parsing limits to maximum 3 questions."""
        response = (
            "1. Question one?\n"
            "2. Question two?\n"
            "3. Question three?\n"
            "4. Question four?\n"
            "5. Question five?"
        )
        questions = generator._parse_questions(response)
        
        assert len(questions) == 3


class TestCandidateFormatting:
    """Test candidate formatting."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        return provider
    
    @pytest.fixture
    def generator(self, mock_provider):
        """Create ClarificationGenerator with mocked provider."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            return ClarificationGenerator()
    
    def test_format_candidates_with_scores(self, generator):
        """Test formatting candidates with scores."""
        candidates = [
            {"procedure_id": "proc1", "procedure_name": "Ignition coil", "combined_score": 0.8},
            {"procedure_id": "proc2", "procedure_name": "Spark plug", "combined_score": 0.7},
            {"procedure_id": "proc3", "procedure_name": "Fuel injector", "combined_score": 0.6}
        ]
        
        formatted = generator._format_candidates(candidates)
        
        assert "Ignition coil" in formatted
        assert "Spark plug" in formatted
        assert "Fuel injector" in formatted
        assert "0.800" in formatted or "0.8" in formatted
        assert "0.700" in formatted or "0.7" in formatted
    
    def test_format_candidates_without_scores(self, generator):
        """Test formatting candidates without scores."""
        candidates = [
            {"procedure_id": "proc1", "procedure_name": "Ignition coil"},
            {"procedure_id": "proc2", "title": "Spark plug"}
        ]
        
        formatted = generator._format_candidates(candidates)
        
        assert "Ignition coil" in formatted
        assert "Spark plug" in formatted
        assert "score" not in formatted.lower()
    
    def test_format_candidates_empty(self, generator):
        """Test formatting empty candidates list."""
        formatted = generator._format_candidates([])
        
        assert "No recommendations" in formatted or "available" in formatted.lower()
    
    def test_format_candidates_max_limit(self, generator):
        """Test that formatting limits to max_candidates."""
        candidates = [
            {"procedure_id": f"proc{i}", "procedure_name": f"Procedure {i}", "combined_score": 0.9 - i*0.1}
            for i in range(5)
        ]
        
        formatted = generator._format_candidates(candidates, max_candidates=3)
        
        # Should only include top 3
        assert "Procedure 0" in formatted
        assert "Procedure 1" in formatted
        assert "Procedure 2" in formatted
        assert "Procedure 3" not in formatted
        assert "Procedure 4" not in formatted
    
    def test_format_candidates_fallback_to_procedure_id(self, generator):
        """Test formatting when procedure_name and title are missing."""
        candidates = [
            {"procedure_id": "proc1", "combined_score": 0.8}
        ]
        
        formatted = generator._format_candidates(candidates)
        
        assert "proc1" in formatted


class TestErrorHandling:
    """Test error handling for various failure scenarios."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        return provider
    
    @pytest.fixture
    def generator(self, mock_provider):
        """Create ClarificationGenerator with mocked provider."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            return ClarificationGenerator()
    
    def test_handle_llm_api_error(self, generator, mock_provider):
        """Test handling of LLM API errors."""
        mock_provider.generate.side_effect = LLMAPIError("API connection failed")
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        # Should return empty list on API error
        assert questions == []
    
    def test_handle_rate_limit_error(self, generator, mock_provider):
        """Test handling of rate limit errors."""
        mock_provider.generate.side_effect = LLMRateLimitError("Rate limit exceeded")
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        # Should return empty list on rate limit error
        assert questions == []
    
    def test_handle_configuration_error(self, generator, mock_provider):
        """Test handling of configuration errors."""
        mock_provider.generate.side_effect = LLMConfigurationError("Invalid API key")
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        # Configuration errors should raise ClarificationGenerationError
        with pytest.raises(ClarificationGenerationError):
            generator.generate_questions(fault_codes, obd_data, ranked_candidates)
    
    def test_handle_unexpected_error(self, generator, mock_provider):
        """Test handling of unexpected errors."""
        mock_provider.generate.side_effect = ValueError("Unexpected error")
        
        fault_codes = ["P0301"]
        obd_data = {"rpm": 2000}
        ranked_candidates = [
            {"procedure_id": "proc1", "procedure_name": "Test", "combined_score": 0.8}
        ]
        
        questions = generator.generate_questions(fault_codes, obd_data, ranked_candidates)
        
        # Should return empty list on unexpected error
        assert questions == []


class TestProviderConfig:
    """Test provider configuration retrieval."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        return provider
    
    @pytest.fixture
    def generator(self, mock_provider):
        """Create ClarificationGenerator with mocked provider."""
        with patch('src.retrieval.clarification_generator.LLMProviderFactory.create_provider', return_value=mock_provider):
            return ClarificationGenerator()
    
    def test_get_provider_config_defaults(self, generator):
        """Test getting provider config with defaults."""
        config = generator._get_provider_config()
        
        assert "temperature" in config
        assert "max_tokens" in config
        assert "timeout" in config
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1000
    
    def test_get_provider_config_custom(self, generator):
        """Test getting provider config with custom values."""
        # Modify generator's llm_config to have custom values
        generator.llm_config["openai"] = {
            "temperature": 0.9,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        config = generator._get_provider_config()
        
        assert config["temperature"] == 0.9
        assert config["max_tokens"] == 2000
        assert config["timeout"] == 60

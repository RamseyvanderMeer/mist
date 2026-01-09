"""
Unit tests for query expansion module.

Tests cover:
- Initialization with default and custom configs
- Query expansion with mocked LLM providers
- Response extraction and cleaning
- Error handling (API errors, rate limits, configuration errors)
- Edge cases and validation
"""
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.retrieval.query_expander import (
    QueryExpander,
    QueryExpansionError,
    QueryExpansionGenerationError
)
from src.llm.provider import (
    LLMProvider,
    LLMAPIError,
    LLMRateLimitError,
    LLMConfigurationError
)
from src.paths import Paths


class TestQueryExpanderExceptions:
    """Test exception hierarchy."""
    
    def test_query_expansion_error(self):
        """Test base exception."""
        with pytest.raises(QueryExpansionError):
            raise QueryExpansionError("Test error")
    
    def test_query_expansion_generation_error(self):
        """Test generation error exception."""
        with pytest.raises(QueryExpansionGenerationError):
            raise QueryExpansionGenerationError("Generation error")
        # Should also be instance of base exception
        assert issubclass(QueryExpansionGenerationError, QueryExpansionError)


class TestQueryExpanderInitialization:
    """Test QueryExpander initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default config path."""
        # Mock LLM provider to avoid needing real API keys
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            expander = QueryExpander()
            assert expander.config_path.exists()
            assert expander.config_path.name == "llm_config.yaml"
            assert expander.prompt_templates is not None
            assert expander.llm_provider is not None
    
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
                    "max_tokens": 500
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
            
            with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
                expander = QueryExpander(config_path=str(config_path))
                assert expander.config_path == config_path
                assert expander.llm_config == config
    
    def test_init_missing_config_file(self):
        """Test initialization with missing config file."""
        with pytest.raises(FileNotFoundError):
            QueryExpander("/nonexistent/path/config.yaml")
    
    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            with pytest.raises(ValueError):
                QueryExpander(str(config_path))
    
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
                QueryExpander(str(config_path))
            assert "providers" in str(exc_info.value).lower()
    
    def test_init_provider_failure(self):
        """Test initialization when all providers fail."""
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', side_effect=RuntimeError("All providers failed")):
            with pytest.raises(RuntimeError) as exc_info:
                QueryExpander()
            assert "All LLM providers failed" in str(exc_info.value)


class TestQueryExpansion:
    """Test query expansion with mocked LLM providers."""
    
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
    def expander(self, mock_provider):
        """Create QueryExpander with mocked provider."""
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            return QueryExpander()
    
    def test_expand_query_basic(self, expander, mock_provider):
        """Test basic query expansion."""
        original_query = "Engine misfire"
        user_responses = ["Happens at idle", "A/C is on"]
        
        mock_provider.generate.return_value = (
            "Engine misfire at idle with A/C system engaged, "
            "diagnostic troubleshooting for misfire conditions"
        )
        
        expanded = expander.expand_query(original_query, user_responses)
        
        assert len(expanded) > len(original_query)
        assert "misfire" in expanded.lower()
        # Verify LLM was called with correct messages
        call_args = mock_provider.generate.call_args
        assert call_args is not None
        messages = call_args.kwargs.get('messages', [])
        assert len(messages) >= 2
        assert "system" in messages[0]["role"]
        assert "user" in messages[1]["role"]
        assert original_query in messages[1]["content"]
    
    def test_expand_query_with_prefix_removal(self, expander, mock_provider):
        """Test query expansion with LLM response containing prefix."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        # LLM might add prefixes like "Expanded query:" or "Here is the expanded query:"
        mock_provider.generate.return_value = (
            "Expanded query: Engine misfire at idle conditions, "
            "diagnostic procedure for cylinder misfire detection"
        )
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should remove the prefix
        assert not expanded.lower().startswith("expanded query:")
        assert "misfire" in expanded.lower()
        assert "idle" in expanded.lower()
    
    def test_expand_query_with_colon_prefix(self, expander, mock_provider):
        """Test query expansion with colon-separated prefix."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        mock_provider.generate.return_value = (
            "Expanded: Engine misfire at idle conditions"
        )
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should remove prefix and colon
        assert not expanded.lower().startswith("expanded:")
        assert "misfire" in expanded.lower()
    
    def test_expand_query_empty_user_responses(self, expander, mock_provider):
        """Test query expansion with empty user responses."""
        original_query = "Engine misfire"
        user_responses = []
        
        mock_provider.generate.return_value = (
            "Engine misfire diagnostic procedure"
        )
        
        expanded = expander.expand_query(original_query, user_responses)
        
        assert len(expanded) > 0
        assert "misfire" in expanded.lower()
        # Verify empty responses are handled
        call_args = mock_provider.generate.call_args
        messages = call_args.kwargs.get('messages', [])
        assert original_query in messages[1]["content"]
    
    def test_expand_query_multiple_user_responses(self, expander, mock_provider):
        """Test query expansion with multiple user responses."""
        original_query = "Engine misfire"
        user_responses = [
            "Happens at idle",
            "A/C is on",
            "Vehicle has 90,000 miles"
        ]
        
        mock_provider.generate.return_value = (
            "Engine misfire at idle with A/C system engaged, "
            "high mileage vehicle diagnostic procedure"
        )
        
        expanded = expander.expand_query(original_query, user_responses)
        
        assert len(expanded) > len(original_query)
        # Verify all responses are included in prompt
        call_args = mock_provider.generate.call_args
        messages = call_args.kwargs.get('messages', [])
        content = messages[1]["content"]
        assert "idle" in content.lower() or "A/C" in content or "90,000" in content
    
    def test_expand_query_preserves_original_on_short_response(self, expander, mock_provider):
        """Test that original query is returned if expanded query is too short."""
        original_query = "Engine misfire diagnostic procedure for cylinder 1"
        user_responses = ["At idle"]
        
        # LLM returns very short response (less than 50% of original)
        mock_provider.generate.return_value = "Misfire"
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should return original query if expanded is too short
        assert expanded == original_query


class TestResponseExtraction:
    """Test response extraction and cleaning."""
    
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
    def expander(self, mock_provider):
        """Create QueryExpander with mocked provider."""
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            return QueryExpander()
    
    def test_extract_expanded_query_clean_response(self, expander):
        """Test extraction with clean response."""
        llm_response = "Engine misfire at idle conditions diagnostic procedure"
        original_query = "Engine misfire"
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        assert expanded == llm_response
        assert "misfire" in expanded.lower()
    
    def test_extract_expanded_query_with_prefix(self, expander):
        """Test extraction with prefix in response."""
        llm_response = "Expanded query: Engine misfire at idle"
        original_query = "Engine misfire"
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        assert not expanded.lower().startswith("expanded query:")
        assert "misfire" in expanded.lower()
    
    def test_extract_expanded_query_with_colon(self, expander):
        """Test extraction with colon after prefix."""
        llm_response = "Expanded: Engine misfire at idle"
        original_query = "Engine misfire"
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        assert not expanded.lower().startswith("expanded:")
        assert not expanded.startswith(":")
        assert "misfire" in expanded.lower()
    
    def test_extract_expanded_query_empty_response(self, expander):
        """Test extraction with empty response."""
        llm_response = ""
        original_query = "Engine misfire"
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        assert expanded == original_query
    
    def test_extract_expanded_query_whitespace_only(self, expander):
        """Test extraction with whitespace-only response."""
        llm_response = "   \n\n   "
        original_query = "Engine misfire"
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        assert expanded == original_query
    
    def test_extract_expanded_query_too_short(self, expander):
        """Test extraction with response that's too short."""
        original_query = "Engine misfire diagnostic procedure for cylinder 1 and 2"
        llm_response = "Misfire"  # Much shorter than original
        
        expanded = expander._extract_expanded_query(llm_response, original_query)
        
        # Should return original if expanded is less than 50% of original length
        assert expanded == original_query
    
    def test_extract_expanded_query_various_prefixes(self, expander):
        """Test extraction with various prefix formats."""
        original_query = "Engine misfire"
        
        prefixes = [
            "expanded query:",
            "expanded:",
            "query:",
            "here is the expanded query:",
            "here's the expanded query:"
        ]
        
        for prefix in prefixes:
            llm_response = f"{prefix} Engine misfire at idle"
            expanded = expander._extract_expanded_query(llm_response, original_query)
            
            assert not expanded.lower().startswith(prefix)
            assert "misfire" in expanded.lower()


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
    def expander(self, mock_provider):
        """Create QueryExpander with mocked provider."""
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            return QueryExpander()
    
    def test_handle_llm_api_error(self, expander, mock_provider):
        """Test handling of LLM API errors."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        mock_provider.generate.side_effect = LLMAPIError("API connection failed")
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should return original query on API error
        assert expanded == original_query
    
    def test_handle_rate_limit_error(self, expander, mock_provider):
        """Test handling of rate limit errors."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        mock_provider.generate.side_effect = LLMRateLimitError("Rate limit exceeded")
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should return original query on rate limit error
        assert expanded == original_query
    
    def test_handle_configuration_error(self, expander, mock_provider):
        """Test handling of configuration errors."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        mock_provider.generate.side_effect = LLMConfigurationError("Invalid API key")
        
        # Configuration errors should raise QueryExpansionGenerationError
        with pytest.raises(QueryExpansionGenerationError):
            expander.expand_query(original_query, user_responses)
    
    def test_handle_unexpected_error(self, expander, mock_provider):
        """Test handling of unexpected errors."""
        original_query = "Engine misfire"
        user_responses = ["At idle"]
        
        mock_provider.generate.side_effect = ValueError("Unexpected error")
        
        expanded = expander.expand_query(original_query, user_responses)
        
        # Should return original query on unexpected error
        assert expanded == original_query


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
    def expander(self, mock_provider):
        """Create QueryExpander with mocked provider."""
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            return QueryExpander()
    
    def test_get_provider_config_defaults(self, expander):
        """Test getting provider config with defaults."""
        config = expander._get_provider_config()
        
        assert "temperature" in config
        assert "max_tokens" in config
        assert "timeout" in config
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1000  # From actual config file
        assert config["timeout"] == 30
    
    def test_get_provider_config_custom(self, expander):
        """Test getting provider config with custom values."""
        # Modify expander's llm_config to have custom values
        expander.llm_config["openai"] = {
            "temperature": 0.9,
            "max_tokens": 1000,
            "timeout": 60
        }
        
        config = expander._get_provider_config()
        
        assert config["temperature"] == 0.9
        assert config["max_tokens"] == 1000
        assert config["timeout"] == 60
    
    def test_get_provider_config_different_provider(self, expander, mock_provider):
        """Test getting provider config for different provider."""
        # Change provider info
        mock_provider.get_model_info.return_value = {
            "provider": "anthropic",
            "model_name": "claude-3-5-sonnet"
        }
        
        expander.llm_config["anthropic"] = {
            "temperature": 0.8,
            "max_tokens": 800,
            "timeout": 45
        }
        
        config = expander._get_provider_config()
        
        assert config["temperature"] == 0.8
        assert config["max_tokens"] == 800
        assert config["timeout"] == 45


class TestIntegrationWithRealConfig:
    """Integration tests with actual project config file."""
    
    def test_loads_actual_config(self):
        """Test that expander loads from actual project config."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            expander = QueryExpander()
            
            # Verify templates are loaded
            assert "query_expansion" in expander.prompt_templates._templates
            
            # Verify structure
            query_expansion = expander.prompt_templates._templates["query_expansion"]
            assert "system" in query_expansion
            assert "user_template" in query_expansion
    
    def test_real_config_query_expansion(self):
        """Test query expansion with real config."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4o"
        }
        
        mock_provider.generate.return_value = (
            "Engine misfire at idle with A/C system engaged, "
            "high mileage vehicle diagnostic procedure"
        )
        
        with patch('src.retrieval.query_expander.LLMProviderFactory.create_provider', return_value=mock_provider):
            expander = QueryExpander()
            
            original_query = "Engine misfire at idle"
            user_responses = [
                "Yes, happens when A/C is on",
                "Vehicle has 90,000 miles"
            ]
            
            expanded = expander.expand_query(original_query, user_responses)
            
            # Verify expansion occurred
            assert len(expanded) > len(original_query)
            assert "misfire" in expanded.lower()
            
            # Verify prompt was formatted correctly
            call_args = mock_provider.generate.call_args
            messages = call_args.kwargs.get('messages', [])
            assert "query expansion" in messages[0]["content"].lower() or "expansion" in messages[0]["content"].lower()
            assert original_query in messages[1]["content"]
            assert "A/C" in messages[1]["content"] or "90,000" in messages[1]["content"]

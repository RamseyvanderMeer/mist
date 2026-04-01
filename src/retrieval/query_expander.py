"""
Query expansion module using LLM to incorporate user clarification responses.

This module provides the QueryExpander class that expands queries with context
from user clarification responses using LLM providers with fallback support.
"""
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import yaml

from src.llm.provider import LLMProviderFactory, LLMProvider, LLMAPIError, LLMRateLimitError, LLMConfigurationError
from src.llm.prompt_templates import PromptTemplates
from src.paths import Paths

logger = logging.getLogger(__name__)


# Exception hierarchy for query expansion errors
class QueryExpansionError(Exception):
    """Base exception for all query expansion errors."""
    pass


class QueryExpansionGenerationError(QueryExpansionError):
    """Exception raised when query expansion fails."""
    pass


class QueryExpander:
    """
    Expands queries with context from user clarification responses using LLM providers.
    
    Uses LLM providers (with fallback support) to expand queries based on original
    query text and user clarification responses.
    
    Attributes:
        config_path: Path to LLM configuration file
        llm_config: Full LLM configuration dictionary
        prompt_templates: PromptTemplates instance for prompt formatting
        llm_provider: LLMProvider instance (with fallback support)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize QueryExpander.
        
        Args:
            config_path: Optional path to LLM config file. If None, uses default
                        from Paths().llm_config.
        
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file is invalid or missing required sections
            RuntimeError: If all LLM providers fail to initialize
        """
        # Determine config path
        if config_path is None:
            paths = Paths()
            config_path = str(paths.llm_config)
        
        self.config_path = Path(config_path)
        
        # Load LLM configuration
        self.llm_config = self._load_config(self.config_path)
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(config_path=self.config_path)
        
        # Initialize LLM provider (with fallback support)
        try:
            self.llm_provider = LLMProviderFactory.create_provider(self.llm_config)
            provider_info = self.llm_provider.get_model_info()
            logger.info(
                f"Initialized QueryExpander with LLM provider: "
                f"{provider_info.get('provider', 'unknown')} "
                f"({provider_info.get('model_name', 'unknown')})"
            )
        except RuntimeError as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise RuntimeError(
                f"All LLM providers failed to initialize. "
                f"Please check your API keys and configuration."
            ) from e
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load LLM configuration from YAML file.
        
        Args:
            config_path: Path to LLM config YAML file
        
        Returns:
            Full LLM configuration dictionary
        
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file is invalid or cannot be parsed
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"LLM config file not found: {config_path}. "
                "Please ensure config/llm_config.yaml exists."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        
        if config is None:
            raise ValueError(f"Config file {config_path} is empty or invalid")
        
        # Validate that providers section exists
        if "providers" not in config:
            raise ValueError(
                f"Missing 'providers' section in config file {config_path}. "
                "Please ensure the config file contains a 'providers' section."
            )
        
        logger.debug(f"Loaded LLM config from {config_path}")
        return config
    
    def expand_query(self, original_query: str, user_responses: List[str]) -> str:
        """
        Expand query with context from user clarification responses.
        
        Uses LLM to generate an expanded query that incorporates the original query
        and user responses in a more comprehensive form suitable for retrieval.
        
        Args:
            original_query: Original query text
            user_responses: List of user clarification response strings
        
        Returns:
            Expanded query text. Returns original query if expansion fails.
        
        Raises:
            QueryExpansionGenerationError: If query expansion fails critically
        """
        try:
            # Get formatted prompt from templates
            prompt = self.prompt_templates.get_query_expansion_prompt(
                original_query=original_query,
                user_responses=user_responses
            )
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            # Get generation parameters from config
            provider_config = self._get_provider_config()
            temperature = provider_config.get("temperature", 0.7)
            max_tokens = provider_config.get("max_tokens", 500)
            
            # Call LLM provider to generate expanded query
            logger.debug("Calling LLM provider to expand query")
            llm_response = self.llm_provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract expanded query from LLM response
            expanded_query = self._extract_expanded_query(llm_response, original_query)
            
            logger.info(f"Successfully expanded query (original length: {len(original_query)}, expanded length: {len(expanded_query)})")
            return expanded_query
        
        except (LLMAPIError, LLMRateLimitError) as e:
            logger.warning(f"LLM API error during query expansion: {e}. Returning original query.")
            return original_query
        except LLMConfigurationError as e:
            logger.error(f"LLM configuration error: {e}")
            raise QueryExpansionGenerationError(
                f"LLM provider configuration error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during query expansion: {e}", exc_info=True)
            return original_query
    
    def _extract_expanded_query(self, llm_response: str, original_query: str) -> str:
        """
        Extract expanded query from LLM response.
        
        Processes the LLM response to extract the expanded query text. If the
        response is empty or invalid, returns the original query.
        
        Args:
            llm_response: Raw LLM response string
            original_query: Original query text (used as fallback)
        
        Returns:
            Expanded query text, or original query if extraction fails
        """
        if not llm_response or not llm_response.strip():
            logger.warning("Empty LLM response received, returning original query")
            return original_query
        
        # Clean up the response
        expanded = llm_response.strip()
        
        # Remove common prefixes/suffixes that LLMs might add
        prefixes_to_remove = [
            "expanded query:",
            "expanded:",
            "query:",
            "here is the expanded query:",
            "here's the expanded query:"
        ]
        for prefix in prefixes_to_remove:
            if expanded.lower().startswith(prefix):
                expanded = expanded[len(prefix):].strip()
                # Remove leading colon if present
                if expanded.startswith(':'):
                    expanded = expanded[1:].strip()
                break
        
        # If the expanded query is too short or seems invalid, return original
        if len(expanded) < len(original_query) * 0.5:
            logger.warning(
                f"Expanded query seems too short ({len(expanded)} chars), "
                f"returning original query"
            )
            return original_query
        
        return expanded

    def expand_symptom_for_search(self, symptom: str) -> str:
        """
        Expand a symptom description into related causes, components, and repairs.

        Bridges the symptom->fix semantic gap: e.g. "engine too hot" ->
        "overheating, coolant system, radiator, thermostat, coolant leak".
        Improves retrieval when user describes symptoms rather than fixes.

        Args:
            symptom: User's problem description (e.g. "engine too hot")

        Returns:
            Expanded text for search, or original symptom if expansion fails
        """
        if not symptom or not symptom.strip():
            return symptom or ""
        symptom = symptom.strip()
        try:
            prompt = self.prompt_templates.get_symptom_expansion_prompt(symptom)
            if not prompt:
                logger.debug("Symptom expansion template not configured, using original")
                return symptom
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
            provider_config = self._get_provider_config()
            llm_response = self.llm_provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=provider_config.get("max_tokens", 150),
            )
            if not llm_response or not llm_response.strip():
                return symptom
            expanded = llm_response.strip()
            # Combine original symptom with expanded terms for richer search
            combined = f"{symptom}, {expanded}" if expanded != symptom else symptom
            logger.info(f"Symptom expansion: '{symptom[:50]}...' -> '{combined[:80]}...'")
            return combined
        except Exception as e:
            logger.warning(f"Symptom expansion failed: {e}. Using original description.")
            return symptom

    def generate_clarifying_questions_for_symptom(
        self,
        symptom_description: str
    ) -> List[str]:
        """
        Generate clarifying questions for a symptom description.

        Uses the mechanic diagnostic question framework to ask targeted questions
        that help narrow down vague symptom descriptions to specific repair guides.

        Args:
            symptom_description: User's symptom/problem description

        Returns:
            List of clarifying questions (1-4 questions)
        """
        if not symptom_description or not symptom_description.strip():
            return []
        
        symptom_description = symptom_description.strip()
        
        try:
            prompt = self.prompt_templates.get_symptom_clarification_prompt(symptom_description)
            if not prompt:
                logger.debug("Symptom clarification template not configured")
                return []
            
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
            
            provider_config = self._get_provider_config()
            llm_response = self.llm_provider.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=provider_config.get("max_tokens", 500),
            )
            
            if not llm_response or not llm_response.strip():
                return []
            
            # Parse questions from response
            questions = self._parse_questions_from_response(llm_response)
            logger.info(f"Generated {len(questions)} clarifying questions for symptom")
            return questions
            
        except Exception as e:
            logger.warning(f"Failed to generate clarifying questions: {e}")
            return []

    def _parse_questions_from_response(self, response: str) -> List[str]:
        """
        Parse numbered or bulleted questions from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of extracted questions
        """
        import re
        
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered questions (1., 1), etc.)
            numbered_match = re.match(r'^\d+[.\)]\s*(.+)', line)
            if numbered_match:
                questions.append(numbered_match.group(1).strip())
                continue
            
            # Match bulleted questions (-, *, •)
            bullet_match = re.match(r'^[-*•]\s*(.+)', line)
            if bullet_match:
                questions.append(bullet_match.group(1).strip())
                continue
            
            # If line ends with question mark, treat as question
            if line.endswith('?'):
                questions.append(line)
        
        return questions

    def _get_provider_config(self) -> Dict[str, Any]:
        """
        Get configuration for the active provider.
        
        Returns:
            Dictionary with provider-specific configuration (temperature, max_tokens, etc.)
        """
        provider_info = self.llm_provider.get_model_info()
        provider_name = provider_info.get("provider", "openai")
        
        # Get provider-specific config from llm_config
        provider_config = self.llm_config.get(provider_name, {})
        
        # Use defaults if not specified
        return {
            "temperature": provider_config.get("temperature", 0.7),
            "max_tokens": provider_config.get("max_tokens", 500),
            "timeout": provider_config.get("timeout", 30)
        }
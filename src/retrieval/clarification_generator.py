"""
Clarification question generation using LLM providers.

This module provides the ClarificationGenerator class that generates clarifying
questions based on fault codes, OBD data, and top-ranked candidates using LLM
providers with fallback support.
"""
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import re
import yaml

from src.llm.provider import LLMProviderFactory, LLMProvider, LLMAPIError, LLMRateLimitError, LLMConfigurationError
from src.llm.prompt_templates import PromptTemplates
from src.paths import Paths

logger = logging.getLogger(__name__)


# Exception hierarchy for clarification generator errors
class ClarificationGeneratorError(Exception):
    """Base exception for all clarification generator errors."""
    pass


class ClarificationGenerationError(ClarificationGeneratorError):
    """Exception raised when question generation fails."""
    pass


class ClarificationGenerator:
    """
    Generates clarifying questions using LLM providers.
    
    Uses LLM providers (with fallback support) to generate 1-3 clarifying
    questions based on fault codes, OBD data, and top-ranked candidates.
    
    Attributes:
        config_path: Path to LLM configuration file
        llm_config: Full LLM configuration dictionary
        prompt_templates: PromptTemplates instance for prompt formatting
        llm_provider: LLMProvider instance (with fallback support)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ClarificationGenerator.
        
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
                f"Initialized ClarificationGenerator with LLM provider: "
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
    
    def generate_questions(
        self,
        fault_codes: List[str],
        obd_data: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate clarifying questions based on fault codes, OBD data, and candidates.
        
        Args:
            fault_codes: List of fault code strings (e.g., ["P0301", "P0302"])
            obd_data: OBD sensor data dictionary (e.g., {"rpm": 2000, "coolant_temp": 90})
            ranked_candidates: List of ranked candidate dictionaries from Ranker.rank().
                            Each dict should contain:
                            - procedure_id: str
                            - procedure_name or title: str (optional)
                            - combined_score: float
                            - Other metadata fields
        
        Returns:
            List of 1-3 clarifying questions as strings. Returns empty list if
            generation fails.
        
        Raises:
            ClarificationGenerationError: If question generation fails critically
        """
        try:
            # Format candidates into string representation
            top_candidates_str = self._format_candidates(ranked_candidates)
            
            # Get formatted prompt from templates
            prompt = self.prompt_templates.get_clarification_prompt(
                fault_codes=fault_codes,
                obd_data=obd_data,
                top_candidates=top_candidates_str
            )
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            # Get generation parameters from config
            provider_config = self._get_provider_config()
            temperature = provider_config.get("temperature", 0.7)
            max_tokens = provider_config.get("max_tokens", 1000)
            
            # Call LLM provider to generate questions
            logger.debug("Calling LLM provider to generate clarification questions")
            llm_response = self.llm_provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse questions from LLM response
            questions = self._parse_questions(llm_response)
            
            logger.info(f"Generated {len(questions)} clarification questions")
            return questions
        
        except (LLMAPIError, LLMRateLimitError) as e:
            logger.error(f"LLM API error during question generation: {e}")
            return []
        except LLMConfigurationError as e:
            logger.error(f"LLM configuration error: {e}")
            raise ClarificationGenerationError(
                f"LLM provider configuration error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during question generation: {e}", exc_info=True)
            return []
    
    def _format_candidates(
        self,
        ranked_candidates: List[Dict[str, Any]],
        max_candidates: int = 3
    ) -> str:
        """
        Format ranked candidates into readable string representation.
        
        Args:
            ranked_candidates: List of ranked candidate dictionaries
            max_candidates: Maximum number of candidates to include (default: 3)
        
        Returns:
            Formatted string describing top candidates
        """
        if not ranked_candidates:
            return "No recommendations available"
        
        # Take top N candidates
        top_n = ranked_candidates[:max_candidates]
        
        formatted_lines = []
        for i, candidate in enumerate(top_n, start=1):
            # Try to get procedure name or title
            procedure_name = (
                candidate.get("procedure_name") or
                candidate.get("title") or
                candidate.get("procedure_id", "Unknown procedure")
            )
            
            # Get combined score if available
            score = candidate.get("combined_score")
            if score is not None:
                formatted_lines.append(
                    f"{i}. {procedure_name} (score: {score:.3f})"
                )
            else:
                formatted_lines.append(f"{i}. {procedure_name}")
        
        return "\n".join(formatted_lines)
    
    def _parse_questions(self, llm_response: str) -> List[str]:
        """
        Parse questions from LLM response.
        
        Handles various formats:
        - Numbered questions (1., 2., 3.)
        - Bulleted questions (-, *, •)
        - Plain text questions
        
        Args:
            llm_response: Raw LLM response string
        
        Returns:
            List of 1-3 parsed questions (cleaned and normalized)
        """
        if not llm_response or not llm_response.strip():
            logger.warning("Empty LLM response received")
            return []
        
        questions = []
        
        # Split response into lines
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match numbered questions (1., 2., 3., etc.)
            numbered_match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if numbered_match:
                question = numbered_match.group(1).strip()
                if question:
                    questions.append(question)
                continue
            
            # Try to match bulleted questions (-, *, •, etc.)
            bulleted_match = re.match(r'^[-*•]\s+(.+)$', line)
            if bulleted_match:
                question = bulleted_match.group(1).strip()
                if question:
                    questions.append(question)
                continue
            
            # If line doesn't match patterns but looks like a question
            # (ends with ? or is substantial text), include it
            if line.endswith('?') or len(line) > 20:
                # Skip if it looks like a header or instruction
                skip_patterns = [
                    'focus on', 'return only', 'generate', 'analyze',
                    'fault codes', 'obd data', 'top recommendations',
                    'here are', 'these questions', 'clarifying questions'
                ]
                if not any(skip in line.lower() for skip in skip_patterns):
                    questions.append(line)
        
        # Limit to 3 questions maximum
        questions = questions[:3]
        
        # Clean up questions (remove extra whitespace, normalize)
        cleaned_questions = []
        for q in questions:
            cleaned = ' '.join(q.split())  # Normalize whitespace
            # Filter out very short fragments and instruction-like text
            skip_patterns = [
                'focus on', 'return only', 'generate', 'analyze',
                'here are', 'these questions', 'clarifying questions'
            ]
            if cleaned and len(cleaned) > 10 and not any(skip in cleaned.lower() for skip in skip_patterns):
                cleaned_questions.append(cleaned)
        
        if not cleaned_questions:
            # Fallback: if no questions were parsed, try to extract sentences
            # ending with question marks or containing question marks
            # Split by sentence boundaries but preserve question marks
            sentences = re.split(r'[.!]+', llm_response)
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for sentences with question marks
                if '?' in sentence:
                    # Extract the part up to and including the question mark
                    parts = sentence.split('?')
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:  # Not the last part
                            question = (part + '?').strip()
                        else:
                            question = part.strip()
                        
                        if question and len(question) > 10:
                            cleaned = ' '.join(question.split())
                            skip_patterns = [
                                'focus on', 'return only', 'generate', 'analyze',
                                'here are', 'these questions', 'clarifying questions'
                            ]
                            if not any(skip in cleaned.lower() for skip in skip_patterns):
                                cleaned_questions.append(cleaned)
                                if len(cleaned_questions) >= 3:
                                    break
                    if len(cleaned_questions) >= 3:
                        break
        
        logger.debug(f"Parsed {len(cleaned_questions)} questions from LLM response")
        return cleaned_questions
    
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
            "max_tokens": provider_config.get("max_tokens", 1000),
            "timeout": provider_config.get("timeout", 30)
        }

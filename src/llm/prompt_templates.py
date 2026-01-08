"""
Prompt templates for clarification and query expansion.

This module provides a PromptTemplates class that loads prompt templates from
YAML configuration files and supports variable substitution.
"""
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.paths import Paths

logger = logging.getLogger(__name__)


class PromptTemplates:
    """
    Manages prompt templates loaded from YAML configuration.
    
    Loads templates from config/llm_config.yaml and provides methods to
    generate prompts with variable substitution.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize PromptTemplates instance.
        
        Args:
            config_path: Optional path to LLM config file. If None, uses
                        Paths().llm_config to get default config path.
        
        Raises:
            FileNotFoundError: If config file does not exist
            KeyError: If prompts section is missing from config
        """
        if config_path is None:
            paths = Paths()
            config_path = paths.llm_config
        
        self.config_path = Path(config_path)
        self._templates = self._load_templates(self.config_path)
    
    def _load_templates(self, config_path: Path) -> Dict[str, Any]:
        """
        Load templates from YAML config file.
        
        Args:
            config_path: Path to LLM config YAML file
        
        Returns:
            Dictionary containing templates from prompts section
        
        Raises:
            FileNotFoundError: If config file does not exist
            KeyError: If prompts section is missing from config
            yaml.YAMLError: If YAML parsing fails
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
        
        if "prompts" not in config:
            raise KeyError(
                f"Missing 'prompts' section in config file {config_path}. "
                "Please ensure the config file contains a 'prompts' section."
            )
        
        prompts = config["prompts"]
        
        # Validate that required templates exist
        required_templates = ["clarification", "query_expansion"]
        missing = [t for t in required_templates if t not in prompts]
        if missing:
            raise KeyError(
                f"Missing required templates in config: {missing}. "
                f"Found templates: {list(prompts.keys())}"
            )
        
        # Validate template structure
        for template_name in required_templates:
            template = prompts[template_name]
            if not isinstance(template, dict):
                raise ValueError(
                    f"Template '{template_name}' must be a dictionary, "
                    f"got {type(template).__name__}"
                )
            if "system" not in template:
                raise KeyError(
                    f"Template '{template_name}' missing required 'system' field"
                )
            if "user_template" not in template:
                raise KeyError(
                    f"Template '{template_name}' missing required 'user_template' field"
                )
        
        logger.info(f"Loaded prompt templates from {config_path}")
        return prompts
    
    def _extract_variables(self, template: str) -> List[str]:
        """
        Extract variable names from template string.
        
        Finds all {variable} placeholders in the template.
        
        Args:
            template: Template string with {variable} placeholders
        
        Returns:
            List of variable names found in template
        """
        # Match {variable} patterns, but not {{escaped}} patterns
        pattern = r'\{([^{}]+)\}'
        matches = re.findall(pattern, template)
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        return unique_matches
    
    def _validate_variables(
        self, 
        template: str, 
        provided_vars: Dict[str, Any]
    ) -> None:
        """
        Validate that all required variables are provided.
        
        Args:
            template: Template string with {variable} placeholders
            provided_vars: Dictionary of provided variables
        
        Raises:
            ValueError: If required variables are missing
        """
        required_vars = self._extract_variables(template)
        missing = [var for var in required_vars if var not in provided_vars]
        
        if missing:
            raise ValueError(
                f"Missing required template variables: {missing}. "
                f"Provided variables: {list(provided_vars.keys())}"
            )
    
    def _substitute_template(
        self, 
        template: str, 
        variables: Dict[str, Any]
    ) -> str:
        """
        Substitute variables in template string.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary mapping variable names to values
        
        Returns:
            Template with variables substituted
        
        Raises:
            ValueError: If required variables are missing
            KeyError: If template contains invalid variable references
        """
        self._validate_variables(template, variables)
        
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Template contains invalid variable reference: {e}. "
                f"Available variables: {list(variables.keys())}"
            ) from e
    
    def get_clarification_prompt(
        self,
        fault_codes: List[str],
        obd_data: Dict,
        top_candidates: str
    ) -> Dict[str, str]:
        """
        Get clarification prompt for LLM.
        
        Formats the clarification template with provided variables and returns
        system and user messages.
        
        Args:
            fault_codes: List of fault code strings
            obd_data: OBD sensor data dictionary
            top_candidates: String describing top recommendation candidates
        
        Returns:
            Dictionary with "system" and "user" keys containing prompt messages
        
        Raises:
            KeyError: If clarification template is missing
            ValueError: If required variables are missing or template substitution fails
        """
        if "clarification" not in self._templates:
            raise KeyError(
                "Clarification template not found in loaded templates. "
                "This should not happen if config validation passed."
            )
        
        template = self._templates["clarification"]
        system_prompt = template["system"]
        user_template = template["user_template"]
        
        # Format variables for substitution
        # Convert fault_codes list to comma-separated string
        fault_codes_str = ', '.join(fault_codes)
        
        # Convert obd_data dict to string representation
        obd_data_str = str(obd_data)
        
        variables = {
            "fault_codes": fault_codes_str,
            "obd_data": obd_data_str,
            "top_candidates": top_candidates
        }
        
        user_prompt = self._substitute_template(user_template, variables)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def get_query_expansion_prompt(
        self,
        original_query: str,
        user_responses: List[str]
    ) -> Dict[str, str]:
        """
        Get query expansion prompt for LLM.
        
        Formats the query expansion template with provided variables and returns
        system and user messages.
        
        Args:
            original_query: Original query text
            user_responses: List of user clarification response strings
        
        Returns:
            Dictionary with "system" and "user" keys containing prompt messages
        
        Raises:
            KeyError: If query_expansion template is missing
            ValueError: If required variables are missing or template substitution fails
        """
        if "query_expansion" not in self._templates:
            raise KeyError(
                "Query expansion template not found in loaded templates. "
                "This should not happen if config validation passed."
            )
        
        template = self._templates["query_expansion"]
        system_prompt = template["system"]
        user_template = template["user_template"]
        
        # Format variables for substitution
        # Convert user_responses list to comma-separated string
        user_responses_str = ', '.join(user_responses)
        
        variables = {
            "original_query": original_query,
            "user_responses": user_responses_str
        }
        
        user_prompt = self._substitute_template(user_template, variables)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }

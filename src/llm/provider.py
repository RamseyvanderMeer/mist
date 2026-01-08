"""
Abstract base class for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from messages.
        
        Args:
            messages: List of message dicts with "role" and "content"
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate text stream from messages.
        
        Args:
            messages: List of message dicts
            **kwargs: Additional generation parameters
        
        Returns:
            Iterator of text chunks
        """
        pass


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(config: dict) -> LLMProvider:
        """
        Create LLM provider from configuration.
        
        Args:
            config: Configuration dict with provider settings
        
        Returns:
            LLMProvider instance
        """
        provider_name = config.get("primary", "openai")
        
        # Try primary provider
        try:
            if provider_name == "openai":
                from .openai_client import OpenAIClient
                return OpenAIClient(config.get("openai", {}))
            elif provider_name == "anthropic":
                from .anthropic_client import AnthropicClient
                return AnthropicClient(config.get("anthropic", {}))
            elif provider_name == "open_source":
                from .open_source_client import OpenSourceClient
                return OpenSourceClient(config.get("open_source", {}))
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name}: {e}")
        
        # Try fallback providers
        fallbacks = config.get("fallback", [])
        for fallback_name in fallbacks:
            try:
                if fallback_name == "openai":
                    from .openai_client import OpenAIClient
                    return OpenAIClient(config.get("openai", {}))
                elif fallback_name == "anthropic":
                    from .anthropic_client import AnthropicClient
                    return AnthropicClient(config.get("anthropic", {}))
                elif fallback_name == "open_source":
                    from .open_source_client import OpenSourceClient
                    return OpenSourceClient(config.get("open_source", {}))
            except Exception as e:
                logger.error(f"Failed to initialize fallback {fallback_name}: {e}")
        
        raise RuntimeError("All LLM providers failed to initialize")

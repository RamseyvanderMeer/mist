"""
Abstract base class for LLM providers.

This module defines the interface contract for LLM providers, including
synchronous and asynchronous text generation, streaming, and model information
retrieval. All LLM provider implementations must inherit from LLMProvider and
implement all abstract methods.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Any, AsyncIterator
import logging

logger = logging.getLogger(__name__)


# Exception hierarchy for LLM provider errors
class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMAPIError(LLMProviderError):
    """Exception raised for API-related errors (network, authentication, etc.)."""
    pass


class LLMRateLimitError(LLMAPIError):
    """Exception raised when rate limit is exceeded."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Exception raised for configuration-related errors."""
    pass


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface contract that all LLM provider implementations
    must follow. It provides both synchronous and asynchronous methods for text
    generation and streaming.
    
    Message Format:
        Messages should be a list of dictionaries with the following structure:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        Supported roles: "system", "user", "assistant"
    
    Error Handling:
        Implementations should raise appropriate exceptions from the LLMProviderError
        hierarchy:
        - LLMAPIError: For API-related errors (network issues, authentication failures)
        - LLMRateLimitError: For rate limiting (should inherit from LLMAPIError)
        - LLMConfigurationError: For configuration issues (missing API keys, invalid settings)
    
    Example:
        ```python
        class MyLLMProvider(LLMProvider):
            def generate(self, messages, **kwargs):
                # Implementation here
                pass
            
            def stream(self, messages, **kwargs):
                # Implementation here
                pass
            
            # ... implement all abstract methods
        ```
    """
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from messages synchronously.
        
        This method takes a list of messages and generates a complete response.
        The method should handle all API communication, error handling, and
        response parsing internally.
        
        Args:
            messages: List of message dictionaries. Each dict must have "role"
                     (str) and "content" (str) keys. Supported roles: "system",
                     "user", "assistant".
            **kwargs: Additional generation parameters. Common parameters include:
                     - temperature: float (0.0-2.0) - Controls randomness
                     - max_tokens: int - Maximum tokens in response
                     - timeout: int - Request timeout in seconds
        
        Returns:
            str: The generated text response.
        
        Raises:
            LLMAPIError: If API call fails (network error, authentication, etc.)
            LLMRateLimitError: If rate limit is exceeded
            LLMConfigurationError: If provider is misconfigured
            NotImplementedError: If not implemented by subclass
        
        Example:
            ```python
            messages = [
                {"role": "user", "content": "What is 2+2?"}
            ]
            response = provider.generate(messages, temperature=0.7)
            ```
        """
        raise NotImplementedError(
            "Subclasses must implement the generate() method"
        )
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Stream text from messages synchronously.
        
        This method generates text incrementally, yielding chunks as they become
        available. Useful for real-time response display and lower latency.
        
        Args:
            messages: List of message dictionaries. Each dict must have "role"
                     (str) and "content" (str) keys. Supported roles: "system",
                     "user", "assistant".
            **kwargs: Additional generation parameters. Common parameters include:
                     - temperature: float (0.0-2.0) - Controls randomness
                     - max_tokens: int - Maximum tokens in response
                     - timeout: int - Request timeout in seconds
        
        Yields:
            str: Text chunks as they are generated.
        
        Raises:
            LLMAPIError: If API call fails (network error, authentication, etc.)
            LLMRateLimitError: If rate limit is exceeded
            LLMConfigurationError: If provider is misconfigured
            NotImplementedError: If not implemented by subclass
        
        Example:
            ```python
            messages = [
                {"role": "user", "content": "Tell me a story"}
            ]
            for chunk in provider.stream(messages):
                print(chunk, end="", flush=True)
            ```
        """
        raise NotImplementedError(
            "Subclasses must implement the stream() method"
        )
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate text stream from messages (legacy method name).
        
        This method is maintained for backward compatibility. New code should
        use stream() instead. By default, this should delegate to stream().
        
        Args:
            messages: List of message dicts with "role" and "content"
            **kwargs: Additional generation parameters
        
        Returns:
            Iterator[str]: Iterator of text chunks
        
        Note:
            This method name is kept for backward compatibility. Implementations
            can simply delegate to stream() or provide their own implementation.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate_stream() method"
        )
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and provider configuration.
        
        Returns metadata about the model being used, including model name,
        provider type, capabilities, and configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information. Should include:
                - model_name: str - Name/identifier of the model
                - provider: str - Provider name (e.g., "openai", "anthropic")
                - max_tokens: int - Maximum tokens supported
                - supports_streaming: bool - Whether streaming is supported
                - supports_async: bool - Whether async operations are supported
                - temperature_range: tuple[float, float] - Valid temperature range
                - Any other provider-specific metadata
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            ```python
            info = provider.get_model_info()
            print(f"Using model: {info['model_name']}")
            print(f"Max tokens: {info['max_tokens']}")
            ```
        """
        raise NotImplementedError(
            "Subclasses must implement the get_model_info() method"
        )
    
    @abstractmethod
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from messages asynchronously.
        
        Async version of generate(). Use this method when working in async contexts
        to avoid blocking the event loop.
        
        Args:
            messages: List of message dictionaries. Each dict must have "role"
                     (str) and "content" (str) keys. Supported roles: "system",
                     "user", "assistant".
            **kwargs: Additional generation parameters. Common parameters include:
                     - temperature: float (0.0-2.0) - Controls randomness
                     - max_tokens: int - Maximum tokens in response
                     - timeout: int - Request timeout in seconds
        
        Returns:
            str: The generated text response.
        
        Raises:
            LLMAPIError: If API call fails (network error, authentication, etc.)
            LLMRateLimitError: If rate limit is exceeded
            LLMConfigurationError: If provider is misconfigured
            NotImplementedError: If not implemented by subclass
        
        Example:
            ```python
            messages = [
                {"role": "user", "content": "What is 2+2?"}
            ]
            response = await provider.agenerate(messages, temperature=0.7)
            ```
        """
        raise NotImplementedError(
            "Subclasses must implement the agenerate() method"
        )
    
    @abstractmethod
    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """
        Stream text from messages asynchronously.
        
        Async version of stream(). Use this method when working in async contexts
        to avoid blocking the event loop while streaming responses.
        
        Args:
            messages: List of message dictionaries. Each dict must have "role"
                     (str) and "content" (str) keys. Supported roles: "system",
                     "user", "assistant".
            **kwargs: Additional generation parameters. Common parameters include:
                     - temperature: float (0.0-2.0) - Controls randomness
                     - max_tokens: int - Maximum tokens in response
                     - timeout: int - Request timeout in seconds
        
        Yields:
            str: Text chunks as they are generated.
        
        Raises:
            LLMAPIError: If API call fails (network error, authentication, etc.)
            LLMRateLimitError: If rate limit is exceeded
            LLMConfigurationError: If provider is misconfigured
            NotImplementedError: If not implemented by subclass
        
        Example:
            ```python
            messages = [
                {"role": "user", "content": "Tell me a story"}
            ]
            async for chunk in provider.astream(messages):
                print(chunk, end="", flush=True)
            ```
        """
        raise NotImplementedError(
            "Subclasses must implement the astream() method"
        )


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
        logger.info(f"Trying primary LLM provider: {provider_name}")
        
        # Try primary provider
        try:
            if provider_name == "openai":
                from .openai_client import OpenAIClient
                return OpenAIClient(config.get("openai", {}))
            elif provider_name == "anthropic":
                from .anthropic_client import AnthropicClient
                return AnthropicClient(config.get("anthropic", {}))
            elif provider_name == "gemini":
                from .gemini_client import GeminiClient
                return GeminiClient(config.get("gemini", {}))
            elif provider_name == "open_source":
                from .open_source_client import OpenSourceClient
                return OpenSourceClient(config.get("open_source", {}))
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try fallback providers
        fallbacks = config.get("fallback", [])
        logger.info(f"Trying fallback providers: {fallbacks}")
        for fallback_name in fallbacks:
            try:
                if fallback_name == "openai":
                    from .openai_client import OpenAIClient
                    return OpenAIClient(config.get("openai", {}))
                elif fallback_name == "anthropic":
                    from .anthropic_client import AnthropicClient
                    return AnthropicClient(config.get("anthropic", {}))
                elif fallback_name == "gemini":
                    from .gemini_client import GeminiClient
                    return GeminiClient(config.get("gemini", {}))
                elif fallback_name == "open_source":
                    from .open_source_client import OpenSourceClient
                    return OpenSourceClient(config.get("open_source", {}))
            except Exception as e:
                logger.error(f"Failed to initialize fallback {fallback_name}: {e}")
        
        raise RuntimeError("All LLM providers failed to initialize")

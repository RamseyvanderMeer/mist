"""
Google Gemini LLM client implementation.
"""
import os
import asyncio
from typing import List, Dict, Iterator, Any, AsyncIterator
import google.generativeai as genai
from .provider import (
    LLMProvider,
    LLMAPIError,
    LLMRateLimitError,
    LLMConfigurationError,
)
import logging

logger = logging.getLogger(__name__)


class GeminiClient(LLMProvider):
    """Google Gemini client"""
    
    def __init__(self, config: dict):
        """
        Initialize Gemini client.
        
        Args:
            config: Configuration dict with model, api_key_env, etc.
        """
        api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        
        # Also check GEMINI_API_KEY directly if not found via config
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise LLMConfigurationError(
                f"Gemini API key not found in environment variable '{api_key_env}' or 'GEMINI_API_KEY'"
            )
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Get model from config or environment variable
        self.model_name = config.get("model") or os.getenv("GEMINI_MODEL", "gemini-pro")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.timeout = config.get("timeout", 30)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to initialize Gemini model '{self.model_name}': {e}"
            )
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Convert standard message format to Gemini format.
        
        Gemini uses:
        - system_instruction: Separate parameter for system messages
        - contents: List of message parts with role "user" or "model"
        
        Args:
            messages: List of message dicts with "role" and "content"
        
        Returns:
            Tuple of (system_instruction, contents_list)
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini uses system_instruction parameter
                if system_instruction is None:
                    system_instruction = content
                else:
                    # Append to existing system instruction
                    system_instruction += "\n\n" + content
            elif role == "user":
                contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [content]})
            else:
                logger.warning(f"Unknown message role: {role}, treating as user message")
                contents.append({"role": "user", "parts": [content]})
        
        return system_instruction, contents
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from messages synchronously.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            **kwargs: Additional generation parameters
        
        Returns:
            str: Generated text response
        
        Raises:
            LLMAPIError: If API call fails
            LLMRateLimitError: If rate limit exceeded
            LLMConfigurationError: If misconfigured
        """
        try:
            system_instruction, contents = self._convert_messages(messages)
            
            # Ensure we have at least one user message
            if not contents:
                raise LLMAPIError("No user messages found in message list")
            
            generation_config = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Handle system instruction by creating a model with system instruction
            # or including it in the first user message
            if system_instruction:
                # Create a new model instance with system instruction
                model_with_system = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model_with_system = self.model
            
            # Start a chat session if we have conversation history
            if len(contents) > 1:
                chat = model_with_system.start_chat(history=contents[:-1])
                response = chat.send_message(
                    contents[-1]["parts"][0],
                    generation_config=generation_config,
                )
            else:
                # Single message - use generate_content
                prompt = contents[0]["parts"][0]
                response = model_with_system.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
            
            # Extract text from response
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    return "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
            
            raise LLMAPIError("Unexpected response format from Gemini API")
            
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                raise LLMRateLimitError(f"Gemini API rate limit exceeded: {e}") from e
            elif "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise LLMConfigurationError(f"Gemini API authentication failed: {e}") from e
            else:
                raise LLMAPIError(f"Gemini API error: {e}") from e
    
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Stream text from messages synchronously.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            **kwargs: Additional generation parameters
        
        Yields:
            str: Text chunks as they are generated
        
        Raises:
            LLMAPIError: If API call fails
            LLMRateLimitError: If rate limit exceeded
            LLMConfigurationError: If misconfigured
        """
        try:
            system_instruction, contents = self._convert_messages(messages)
            
            # Ensure we have at least one user message
            if not contents:
                raise LLMAPIError("No user messages found in message list")
            
            generation_config = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Handle system instruction by creating a model with system instruction
            if system_instruction:
                # Create a new model instance with system instruction
                model_with_system = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model_with_system = self.model
            
            # Start a chat session if we have conversation history
            if len(contents) > 1:
                chat = model_with_system.start_chat(history=contents[:-1])
                response_stream = chat.send_message(
                    contents[-1]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )
            else:
                # Single message - use generate_content with stream
                prompt = contents[0]["parts"][0]
                response_stream = model_with_system.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True,
                )
            
            # Yield text chunks
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                yield part.text
                
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                raise LLMRateLimitError(f"Gemini API rate limit exceeded: {e}") from e
            elif "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise LLMConfigurationError(f"Gemini API authentication failed: {e}") from e
            else:
                raise LLMAPIError(f"Gemini API error: {e}") from e
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate text stream from messages (legacy method name).
        
        Delegates to stream() for backward compatibility.
        """
        return self.stream(messages, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and provider configuration.
        
        Returns:
            Dict containing model information
        """
        return {
            "model_name": self.model_name,
            "provider": "gemini",
            "max_tokens": self.max_tokens,
            "supports_streaming": True,
            "supports_async": True,
            "temperature_range": (0.0, 2.0),
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text from messages asynchronously.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            **kwargs: Additional generation parameters
        
        Returns:
            str: Generated text response
        
        Raises:
            LLMAPIError: If API call fails
            LLMRateLimitError: If rate limit exceeded
            LLMConfigurationError: If misconfigured
        """
        # Run synchronous generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, messages, **kwargs)
    
    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """
        Stream text from messages asynchronously.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            **kwargs: Additional generation parameters
        
        Yields:
            str: Text chunks as they are generated
        
        Raises:
            LLMAPIError: If API call fails
            LLMRateLimitError: If rate limit exceeded
            LLMConfigurationError: If misconfigured
        """
        # Run synchronous stream in executor and yield chunks asynchronously
        loop = asyncio.get_event_loop()
        stream = self.stream(messages, **kwargs)
        
        # Convert sync iterator to async iterator
        # Use a queue-based approach to avoid blocking
        def get_next_chunk():
            try:
                return next(stream)
            except StopIteration:
                return None
        
        while True:
            chunk = await loop.run_in_executor(None, get_next_chunk)
            if chunk is None:
                break
            yield chunk

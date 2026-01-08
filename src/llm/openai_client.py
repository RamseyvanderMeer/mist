"""
OpenAI LLM client implementation.
"""
import os
from typing import List, Dict, Iterator
from openai import OpenAI
from .provider import LLMProvider
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(LLMProvider):
    """OpenAI GPT-4/GPT-4o client"""
    
    def __init__(self, config: dict):
        """
        Initialize OpenAI client.
        
        Args:
            config: Configuration dict with model, api_key_env, etc.
        """
        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        return response.choices[0].message.content
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Generate streaming text"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

"""
Open-source LLM client (Ollama) implementation.
"""
import os
from typing import List, Dict, Iterator
from langchain_community.llms import Ollama
from .provider import LLMProvider
import logging

logger = logging.getLogger(__name__)


class OpenSourceClient(LLMProvider):
    """Open-source LLM client (Ollama)"""
    
    def __init__(self, config: dict):
        """
        Initialize open-source client.
        
        Args:
            config: Configuration dict with provider, model, base_url, etc.
        """
        self.provider = config.get("provider", "ollama")
        self.model_name = config.get("model", "llama3.1:70b")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        if self.provider == "ollama":
            self.client = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text"""
        prompt = self._convert_messages(messages)
        return self.client.invoke(prompt)
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Generate streaming text"""
        prompt = self._convert_messages(messages)
        for chunk in self.client.stream(prompt):
            yield chunk

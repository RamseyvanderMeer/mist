"""
Anthropic Claude LLM client implementation.
"""
import os
from typing import List, Dict, Iterator
from anthropic import Anthropic
from .provider import LLMProvider
import logging

logger = logging.getLogger(__name__)


class AnthropicClient(LLMProvider):
    """Anthropic Claude client"""
    
    def __init__(self, config: dict):
        """
        Initialize Anthropic client.
        
        Args:
            config: Configuration dict with model, api_key_env, etc.
        """
        api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        self.client = Anthropic(api_key=api_key)
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert standard format to Anthropic format"""
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        return system_message, anthropic_messages
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text"""
        system_msg, anthropic_msgs = self._convert_messages(messages)
        
        response = self.client.messages.create(
            model=self.model,
            system=system_msg,
            messages=anthropic_msgs,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        
        return response.content[0].text
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Generate streaming text"""
        system_msg, anthropic_msgs = self._convert_messages(messages)
        
        with self.client.messages.stream(
            model=self.model,
            system=system_msg,
            messages=anthropic_msgs,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        ) as stream:
            for text in stream.text_stream:
                yield text

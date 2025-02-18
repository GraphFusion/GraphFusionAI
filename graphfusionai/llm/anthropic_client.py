import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from typing import Optional, Dict, Any
import anthropic

class AnthropicClient:
    """
    Client for interacting with Anthropic's Claude LLM API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY env var")
        
        self.client = anthropic.Client(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 1000, 
                temperature: float = 0.7, model: str = "claude-2",
                system_prompt: Optional[str] = None) -> str:
        """
        Generate text completion using Claude.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            model: Model to use (e.g. "claude-2")
            system_prompt: Optional system prompt to prepend

        Returns:
            Generated text completion
        """
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        messages.append({
            "role": "user", 
            "content": prompt
        })

        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.content[0].text

    def stream_generate(self, prompt: str, max_tokens: int = 1000,
                       temperature: float = 0.7, model: str = "claude-2",
                       system_prompt: Optional[str] = None):
        """
        Stream text completion using Claude.

        Args:
            prompt: Input prompt text  
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            model: Model to use (e.g. "claude-2")
            system_prompt: Optional system prompt to prepend

        Yields:
            Generated text chunks as they become available
        """
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        messages.append({
            "role": "user",
            "content": prompt
        })

        with self.client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        ) as stream:
            for chunk in stream:
                if chunk.content:
                    yield chunk.content[0].text

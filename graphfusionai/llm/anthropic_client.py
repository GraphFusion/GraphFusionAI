import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from typing import Optional, Dict, Any, List, Generator
import anthropic
from .base_llm import BaseLLM

class AnthropicClient(BaseLLM):
    """
    Client for interacting with Anthropic's Claude LLM API.
    """

    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.

        Args:
            model: Model to use (e.g. "claude-3-sonnet-20240229", "claude-3-opus-20240229")
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var
        """
        super().__init__(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY env var")
        
        self.client = anthropic.Client(api_key=self.api_key)
        
        # Set context window based on model
        if "opus" in model:
            self.context_window = 200000  # Claude 3 Opus
        elif "sonnet" in model:
            self.context_window = 200000  # Claude 3 Sonnet
        else:
            self.context_window = 100000  # Default for other models

    def call(self, messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Make a call to the Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            Generated text response
        """
        api_messages = []
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        api_messages.extend(messages)

        response = self.client.messages.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return response.content[0].text

    def stream(self, messages: List[Dict[str, str]],
               max_tokens: Optional[int] = None,
               temperature: float = 0.7,
               system_prompt: Optional[str] = None,
               **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream responses from the Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional Anthropic-specific parameters

        Yields:
            Generated text chunks as they become available
        """
        api_messages = []
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        api_messages.extend(messages)

        with self.client.messages.stream(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ) as stream:
            for chunk in stream:
                if chunk.content:
                    yield chunk.content[0].text
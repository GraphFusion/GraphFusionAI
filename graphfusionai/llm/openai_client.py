import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from typing import Optional, Dict, Any, Generator, List
import openai
from .base_llm import BaseLLM

class OpenAIClient(BaseLLM):
    """
    OpenAI implementation of the BaseLLM interface.
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            model: Model to use (e.g. "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
        """
        super().__init__(model)
        
        # Set context window based on model
        if "gpt-4" in model:
            self.context_window = 8192
        else:
            self.context_window = 4096
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
        
        self.client = openai.Client(api_key=self.api_key)

    def call(self, messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Make a call to the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional OpenAI-specific parameters

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return response.choices[0].message.content

    def stream(self, messages: List[Dict[str, str]],
               max_tokens: Optional[int] = None,
               temperature: float = 0.7,
               system_prompt: Optional[str] = None,
               **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream responses from the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional OpenAI-specific parameters

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

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
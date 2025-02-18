import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator, Any, Union

class BaseLLM(ABC):
    """
    Abstract base class for Language Model implementations.
    Provides a standard interface for different LLM providers.
    """
    
    def __init__(self, model: str):
        """
        Initialize the LLM.

        Args:
            model: Name of the model to use
        """
        self.model = model
        self.context_window = 4096  # Default context window size
        self.context_window_margin = 0.75  # Use 75% of context window for safety
        
    def get_context_window(self) -> int:
        """
        Get the safe context window size (with margin applied).

        Returns:
            Integer representing safe token limit
        """
        return int(self.context_window * self.context_window_margin)
    
    @abstractmethod
    def call(self, messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Make a call to the LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]],
               max_tokens: Optional[int] = None,
               temperature: float = 0.7,
               system_prompt: Optional[str] = None,
               **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream responses from the LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Yields:
            Generated text chunks as they become available
        """
        pass

    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                system_prompt: Optional[str] = None,
                **kwargs: Any) -> str:
        """
        Convenience method to generate text from a single prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text completion
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call(messages, max_tokens, temperature, system_prompt, **kwargs)

    def stream_generate(self, prompt: str,
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7,
                       system_prompt: Optional[str] = None,
                       **kwargs: Any) -> Generator[str, None, None]:
        """
        Convenience method to stream text from a single prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Yields:
            Generated text chunks as they become available
        """
        messages = [{"role": "user", "content": prompt}]
        yield from self.stream(messages, max_tokens, temperature, system_prompt, **kwargs)
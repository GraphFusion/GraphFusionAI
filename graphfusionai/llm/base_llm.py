from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator, Any, Union, Tuple
from enum import Enum
import logging
import time
from datetime import datetime

class ModelRole(Enum):
    """Enum for different message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class LLMError(Exception):
    """Base exception class for LLM-related errors"""
    pass

class TokenLimitError(LLMError):
    """Raised when input exceeds model's token limit"""
    pass

class APIError(LLMError):
    """Raised when API call fails"""
    pass

class BaseLLM(ABC):
    """
    Enhanced abstract base class for Language Model implementations.
    Provides a standard interface and common utilities for different LLM providers.
    """
    
    def __init__(self, 
                 model: str,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 30.0):
        """
        Initialize the LLM.

        Args:
            model: Name of the model to use
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Delay between retries in seconds
            timeout: Timeout for API calls in seconds
        """
        self.model = model
        self.context_window = 4096  
        self.context_window_margin = 0.75  
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        self.total_tokens_used = 0
        self.request_count = 0
        self.last_request_time = None
        
    def get_context_window(self) -> int:
        """
        Get the safe context window size (with margin applied).

        Returns:
            Integer representing safe token limit
        """
        return int(self.context_window * self.context_window_margin)
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate message format and content.

        Args:
            messages: List of message dictionaries to validate

        Raises:
            ValueError: If messages are invalid
        """
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Messages must contain 'role' and 'content' keys")
            if not isinstance(msg["content"], str):
                raise ValueError("Message content must be a string")
            if msg["role"] not in [role.value for role in ModelRole]:
                raise ValueError(f"Invalid role: {msg['role']}")

    def _track_usage(self, tokens_used: int) -> None:
        """
        Track token usage and request metrics.

        Args:
            tokens_used: Number of tokens used in the request
        """
        self.total_tokens_used += tokens_used
        self.request_count += 1
        self.last_request_time = datetime.now()

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary containing usage metrics
        """
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "last_request": self.last_request_time,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count if self.request_count > 0 else 0
            )
        }

    @abstractmethod
    def call(self, 
             messages: List[Dict[str, str]], 
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

        Raises:
            LLMError: Base class for LLM-related errors
            TokenLimitError: If input exceeds model's token limit
            APIError: If API call fails
        """
        pass
    
    @abstractmethod
    def stream(self,
               messages: List[Dict[str, str]],
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

        Raises:
            LLMError: Base class for LLM-related errors
            TokenLimitError: If input exceeds model's token limit
            APIError: If API call fails
        """
        pass

    def generate(self, 
                prompt: str, 
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
        messages = [{"role": ModelRole.USER.value, "content": prompt}]
        return self.call(messages, max_tokens, temperature, system_prompt, **kwargs)

    def stream_generate(self,
                       prompt: str,
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
        messages = [{"role": ModelRole.USER.value, "content": prompt}]
        yield from self.stream(messages, max_tokens, temperature, system_prompt, **kwargs)
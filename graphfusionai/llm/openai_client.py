import os
import time
from typing import Optional, Dict, Any, Generator, List
import openai
from openai import OpenAIError
from .base_llm import BaseLLM, LLMError, TokenLimitError, APIError, ModelRole

class OpenAIClient(BaseLLM):
    """
    Enhanced OpenAI implementation of the BaseLLM interface.
    """

    def __init__(self, 
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 30.0):
        """
        Initialize the OpenAI client.

        Args:
            model: Model to use (e.g. "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Timeout for API calls in seconds
        """
        super().__init__(model, max_retries, retry_delay, timeout)
        
        # Set context window based on model
        if "gpt-4" in model:
            self.context_window = 8192
        else:
            self.context_window = 4096
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
        
        self.client = openai.Client(
            api_key=self.api_key,
            timeout=timeout
        )

    def _handle_api_error(self, e: Exception) -> None:
        """Handle OpenAI API errors and raise appropriate exceptions."""
        if "context_length_exceeded" in str(e):
            raise TokenLimitError(f"Input exceeds model's token limit: {str(e)}")
        else:
            raise APIError(f"OpenAI API error: {str(e)}")

    def call(self, 
             messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Make a call to the OpenAI API with retry logic and error handling.
        """
        self._validate_messages(messages)
        
        api_messages = []
        if system_prompt:
            api_messages.append({
                "role": ModelRole.SYSTEM.value,
                "content": system_prompt
            })
        api_messages.extend(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                # Track usage
                self._track_usage(response.usage.total_tokens)
                
                return response.choices[0].message.content

            except OpenAIError as e:
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    self._handle_api_error(e)
                time.sleep(self.retry_delay)

    def stream(self,
               messages: List[Dict[str, str]],
               max_tokens: Optional[int] = None,
               temperature: float = 0.7,
               system_prompt: Optional[str] = None,
               **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream responses from the OpenAI API with retry logic and error handling.
        """
        self._validate_messages(messages)
        
        api_messages = []
        if system_prompt:
            api_messages.append({
                "role": ModelRole.SYSTEM.value,
                "content": system_prompt
            })
        api_messages.extend(messages)

        total_tokens = 0
        for attempt in range(self.max_retries):
            try:
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
                        if hasattr(chunk, 'usage'):
                            total_tokens += chunk.usage.total_tokens
                        yield chunk.choices[0].delta.content

                # Track usage after successful completion
                if total_tokens > 0:
                    self._track_usage(total_tokens)
                break

            except OpenAIError as e:
                self.logger.warning(f"Stream failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    self._handle_api_error(e)
                time.sleep(self.retry_delay)
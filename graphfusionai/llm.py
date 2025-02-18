import json
import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Generator

from dotenv import load_dotenv
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import litellm
    from litellm import get_supported_openai_params

load_dotenv()

# Context window sizes for different models
LLM_CONTEXT_WINDOW_SIZES = {
    # openai
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16385,
    # anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    # gemini
    "gemini-pro": 32768,
    "gemini-1.5-pro": 2097152,
    # mistral
    "mistral-tiny": 32768,
    "mistral-small": 32768,
    "mistral-medium": 32768,
    # other providers...
}

DEFAULT_CONTEXT_WINDOW_SIZE = 8192
CONTEXT_WINDOW_USAGE_RATIO = 0.75

class FilteredStream:
    """Filter out unwanted messages from LiteLLM output streams."""
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s) -> int:
        with self._lock:
            # Filter out extraneous messages
            if any(msg in s for msg in [
                "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new",
                "LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True`"
            ]):
                return 0
            return self._original_stream.write(s)

    def flush(self):
        with self._lock:
            return self._original_stream.flush()

@contextmanager
def suppress_warnings():
    """Context manager to suppress warnings and filter streams."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore", message="open_text is deprecated*", category=DeprecationWarning
        )
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = FilteredStream(old_stdout)
        sys.stderr = FilteredStream(old_stderr)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class LiteLLMClient:
    """Unified LLM client using LiteLLM to support multiple providers."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        timeout: Optional[Union[float, int]] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        callbacks: List[Any] = None,
        **kwargs: Any
    ):
        """
        Initialize the LLM client.

        Args:
            model: Name of the model to use (e.g., "gpt-4", "claude-3-sonnet")
            temperature: Sampling temperature (0.0 to 1.0)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            api_key: API key for the provider
            base_url: Base URL for API requests
            api_base: Alternative base URL
            api_version: API version to use
            callbacks: List of callback functions
            **kwargs: Additional model-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.api_base = api_base
        self.api_version = api_version
        self.callbacks = callbacks or []
        self.additional_params = kwargs
        
        # Set context window based on model
        self.context_window = self._get_context_window_size()
        
        # Configure LiteLLM
        litellm.drop_params = True
        self._setup_callbacks()
        
        # Set provider-specific configurations
        self._setup_provider()

    def _setup_provider(self):
        """Configure provider-specific settings."""
        if self._is_anthropic_model():
            if not self.api_key:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif "gpt" in self.model.lower():
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
        elif "gemini" in self.model.lower():
            if not self.api_key:
                self.api_key = os.getenv("GOOGLE_API_KEY")
                
        if not self.api_key:
            raise ValueError(f"No API key found for model {self.model}")

    def _is_anthropic_model(self) -> bool:
        """Check if the model is from Anthropic."""
        ANTHROPIC_PREFIXES = ('anthropic/', 'claude-', 'claude/')
        return any(prefix in self.model.lower() for prefix in ANTHROPIC_PREFIXES)

    def _get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        for model_prefix, window_size in LLM_CONTEXT_WINDOW_SIZES.items():
            if self.model.startswith(model_prefix):
                return int(window_size * CONTEXT_WINDOW_USAGE_RATIO)
        return int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)

    def _setup_callbacks(self):
        """Set up LiteLLM callbacks."""
        with suppress_warnings():
            # Clear existing callbacks
            litellm.success_callback = []
            litellm._async_success_callback = []
            
            # Add instance callbacks
            if self.callbacks:
                litellm.callbacks = self.callbacks
            
            # Add environment-configured callbacks
            success_callbacks = os.getenv("LITELLM_SUCCESS_CALLBACKS", "").split(",")
            failure_callbacks = os.getenv("LITELLM_FAILURE_CALLBACKS", "").split(",")
            
            if success_callbacks[0]:
                litellm.success_callback.extend(success_callbacks)
            if failure_callbacks[0]:
                litellm.failure_callback.extend(failure_callbacks)

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> str:
        """
        Make a call to the LLM.

        Args:
            messages: String prompt or list of message dictionaries
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional call-specific parameters

        Returns:
            Generated text response
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = self._format_messages(messages)

        with suppress_warnings():
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    tools=tools,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    **{**self.additional_params, **kwargs}
                )
                
                return response.choices[0].message.content or ""
                
            except Exception as e:
                logging.error(f"LiteLLM call failed: {str(e)}")
                raise

    def stream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Stream responses from the LLM.

        Args:
            messages: String prompt or list of message dictionaries
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional call-specific parameters

        Yields:
            Generated text chunks
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = self._format_messages(messages)

        with suppress_warnings():
            try:
                stream = litellm.completion(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    tools=tools,
                    stream=True,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    **{**self.additional_params, **kwargs}
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                logging.error(f"LiteLLM streaming failed: {str(e)}")
                raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages according to provider requirements."""
        if self._is_anthropic_model() and messages and messages[0]["role"] == "system":
            return [{"role": "user", "content": "."}, *messages]
        return messages

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        try:
            params = get_supported_openai_params(model=self.model)
            return "response_format" in params
        except Exception:
            return False

    def get_context_window(self) -> int:
        """Get the safe context window size."""
        return self.context_window
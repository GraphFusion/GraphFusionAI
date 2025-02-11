import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import openai
from typing import List, Dict, Optional
from .base_llm import BaseLLM
import logging

logger = logging.getLogger(__name__)

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        """
        Initialize the OpenAIClient with API key and model.

        Args:
            api_key (str): OpenAI API key.
            model (str): Model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        super().__init__(api_key, model)
        self.context_limits = self._fetch_context_limits()

    def _fetch_context_limits(self) -> Dict[str, int]:
        """
        Returns the context window size for supported models.

        Returns:
            Dict[str, int]: Mapping of model names to context window sizes.
        """
        return {
            "gpt-4": 8192,
            "gpt-4-turbo": 8192,
            "gpt-3.5-turbo": 4096,
        }

    def get_context_window_size(self) -> int:
        """
        Get the context window size for the selected model.

        Returns:
            int: The context window size.
        """
        return self.context_limits.get(self.model, 4096)

    def call(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Calls the OpenAI API with the given messages.

        Args:
            messages (List[Dict[str, str]]): Conversation messages in OpenAI format.
            kwargs: Additional parameters (e.g., temperature, max_tokens).

        Returns:
            Optional[str]: The response text, or None if an error occurs.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                api_key=self.api_key,  # Avoid setting global state
                **kwargs
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None  # Returning None instead of raising an error
        except Exception as e:
            logger.exception(f"Unexpected error in OpenAIClient: {str(e)}")
            return None

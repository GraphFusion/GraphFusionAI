import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import openai
from typing import List, Dict
from .base_llm import BaseLLM

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        openai.api_key = self.api_key

    def call(self, messages, **kwargs):
        """Call OpenAI API with messages."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def get_context_window_size(self):
        context_limits = {
            "gpt-4": 8192,
            "gpt-4-turbo": 8192,
            "gpt-3.5-turbo": 4096,
        }
        return context_limits.get(self.model, 4096)

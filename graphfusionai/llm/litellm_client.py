

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict
import litellm
from .base_llm import BaseLLM

class LiteLLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.api_key = api_key

    def call(self, messages: List[Dict[str, str]]) -> str:
        params = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
        }
        response = litellm.completion(**params)
        return response["choices"][0]["message"]["content"]
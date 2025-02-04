import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from .base_llm import BaseLLM

class AnthropicClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = Anthropic(api_key=api_key)

    def call(self, messages, **kwargs):
        """Call Anthropic API with a formatted prompt."""
        human_prompt = " ".join(
            f"{HUMAN_PROMPT} {msg['content']}" if msg["role"] == "user" else f"{AI_PROMPT} {msg['content']}"
            for msg in messages
        )

        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=human_prompt,
                **kwargs,
            )
            return response.completion.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def get_context_window_size(self):
        context_limits = {
            "claude-v1": 9000,
            "claude-v1.3": 9000,
            "claude-instant": 6000,
        }
        return context_limits.get(self.model, 6000)

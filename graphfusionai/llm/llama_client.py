import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_llm import BaseLLM


class LLaMAClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)

    def call(self, messages, **kwargs):
        """Generate text from LLaMA model using input messages."""
        try:
            prompt = "\n".join([msg["content"] for msg in messages])
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs['input_ids'], max_length=512, **kwargs)
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response_text

        except Exception as e:
            raise RuntimeError(f"LLaMA API error: {str(e)}")

    def get_context_window_size(self):
        """Return LLaMA's context window size, which may vary by model"""
        context_limits = {
            "facebook/llama-7b": 4096,
            "facebook/llama-13b": 4096,
            "facebook/llama-30b": 4096,
        }
        return context_limits.get(self.model, 4096)

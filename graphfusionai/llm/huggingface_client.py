
from typing import List, Dict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .base_llm import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model: str, device: str = "cpu"):
        super().__init__(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def call(self, messages: List[Dict[str, str]]) -> str:
        input_text = " ".join([msg["content"] for msg in messages])
        result = self.pipeline(input_text, max_length=self.get_context_window())[0]
        return result["generated_text"]
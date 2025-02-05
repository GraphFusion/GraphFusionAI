from typing import List, Dict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model: str, device: str = "cpu", max_length: int = 100, temperature: float = 0.7):
        """
        Initializes the HuggingFaceLLM class with model and device.

        Args:
            model (str): Model name from Hugging Face Hub.
            device (str): Device to run the model ('cpu' or 'cuda').
            max_length (int): Maximum length of generated text.
            temperature (float): Sampling temperature. Lower is more deterministic, higher is more random.
        """
        super().__init__(model)\

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self.pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )
        self.max_length = max_length
        self.temperature = temperature

    def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Calls the model to generate a response based on the input messages.

        Args:
            messages (List[Dict[str, str]]): List of message dicts where each contains 'content'.

        Returns:
            str: Generated text based on input messages.
        """
        try:
            input_text = " ".join([msg["content"] for msg in messages])
            response = self.pipeline(
                input_text,
                max_length=self.max_length,
                temperature=self.temperature,
                num_return_sequences=1,  
                truncation=True  
            )
            return response[0]["generated_text"].strip()
        
        except Exception as e:
            
            print(f"Error generating text with HuggingFaceLLM: {e}")
            return "Error generating response."

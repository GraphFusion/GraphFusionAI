from typing import List, Dict, Optional, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def __init__(self, 
                 model: str,
                 device: str = "cpu",
                 max_length: int = 100,
                 temperature: float = 0.7):
        """
        Initialize HuggingFace language model.

        Args:
            model: Model name from Hugging Face Hub
            device: Device to run model on ('cpu' or 'cuda')
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
        """
        super().__init__(model)
        self._initialize_model(model, device)
        self.max_length = max_length
        self.temperature = temperature

    def _initialize_model(self, model: str, device: str) -> None:
        """Initialize model components."""
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def _format_input(self, messages: List[Dict[str, str]]) -> str:
        """Format input messages into single string."""
        return " ".join(msg["content"] for msg in messages)

    def call(self,
             messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None,
             temperature: Optional[float] = None,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Generate text response from input messages.

        Args:
            messages: List of message dicts with 'content' key
            max_tokens: Override default max length if provided
            temperature: Override default temperature if provided
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        try:
            input_text = self._format_input(messages)
            if system_prompt:
                input_text = f"{system_prompt}\n{input_text}"

            response = self.pipeline(
                input_text,
                max_length=max_tokens or self.max_length,
                temperature=temperature or self.temperature,
                num_return_sequences=1,
                truncation=True,
                **kwargs
            )
            return response[0]["generated_text"].strip()

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            return "Error generating response."

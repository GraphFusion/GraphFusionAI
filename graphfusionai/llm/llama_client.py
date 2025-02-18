import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict, Optional, Any, Generator
from llama_cpp import Llama
from .base_llm import BaseLLM


class LlamaClient(BaseLLM):
    """
    Client for interacting with local Llama models using llama-cpp-python.
    """

    def __init__(self, 
                 model: str,
                 model_path: Optional[str] = None,
                 n_ctx: int = 4096,
                 n_gpu_layers: int = 0,
                 **kwargs: Any):
        """
        Initialize the Llama client.

        Args:
            model: Model identifier (e.g., "llama-2-7b-chat")
            model_path: Path to the model file (.gguf format)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            **kwargs: Additional Llama.cpp parameters
        """
        super().__init__(model)
        
        if model_path is None:
            model_path = os.getenv("LLAMA_MODEL_PATH")
            if not model_path:
                raise ValueError("Model path must be provided or set in LLAMA_MODEL_PATH env var")

        self.context_window = n_ctx
        self.model_path = model_path
        
        # Initialize the Llama model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            **kwargs
        )

    def _format_messages(self, 
                        messages: List[Dict[str, str]], 
                        system_prompt: Optional[str] = None) -> str:
        """
        Format messages into Llama-friendly prompt format.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        formatted_prompt = ""
        
        if system_prompt:
            formatted_prompt += f"[SYSTEM] {system_prompt}\n\n"
            
        for message in messages:
            role = message.get("role", "user")
            content = message["content"]
            
            if role == "user":
                formatted_prompt += f"[USER] {content}\n"
            elif role == "assistant":
                formatted_prompt += f"[ASSISTANT] {content}\n"
            
        formatted_prompt += "[ASSISTANT] "
        return formatted_prompt

    def call(self, 
             messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None,
             **kwargs: Any) -> str:
        """
        Generate a response using the Llama model.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        prompt = self._format_messages(messages, system_prompt)
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens or 256,
            temperature=temperature,
            **kwargs
        )
        
        return response["choices"][0]["text"].strip()

    def stream(self, 
               messages: List[Dict[str, str]],
               max_tokens: Optional[int] = None,
               temperature: float = 0.7,
               system_prompt: Optional[str] = None,
               **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream responses from the Llama model.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        prompt = self._format_messages(messages, system_prompt)
        
        stream = self.llm(
            prompt,
            max_tokens=max_tokens or 256,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk["choices"][0]["text"]:
                yield chunk["choices"][0]["text"]

    def get_context_window(self) -> int:
        """
        Get the usable context window size.

        Returns:
            Safe context window size with margin applied
        """
        return int(self.context_window * self.context_window_margin)

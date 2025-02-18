import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseLLM(ABC):
    def __init__(self, model: str):
        """Initialize base LLM class.
        
        Args:
            model: Name/identifier of the model to use
        """
        self.model = model
        self.context_window = 8192
        
    @abstractmethod 
    def call(self, messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: float = 0.7,
             system_prompt: Optional[str] = None) -> str:
        """Make a call to the LLM with given input messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature between 0 and 1
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated text response
        """
        pass

    def get_context_window(self) -> int:
        """Return the usable context window size.
        
        Returns:
            Effective context window size (75% of full window)
        """
        return int(self.context_window * 0.75)

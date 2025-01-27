from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLM(ABC):
    def __init__(self, model: str):
        self.model = model
        self.context_window = 8192  
    @abstractmethod
    def call(self, messages: List[Dict[str, str]]) -> str:
        """Make a call to the LLM with given input messages."""
        pass

    def get_context_window(self) -> int:
        """Return the usable context window size."""
        return int(self.context_window * 0.75)


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .huggingface_client import HuggingFaceLLM
from .litellm_client import LiteLLM
from .openai_client import OpenAIClient

def create_llm(provider: str, model: str, **kwargs):
    if provider == "huggingface":
        return HuggingFaceLLM(model, **kwargs)
    elif provider == "litellm":
        return LiteLLM(model, **kwargs)
    if provider == "openai":
        return OpenAIClient(model, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

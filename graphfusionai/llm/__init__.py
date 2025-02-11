
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .huggingface_client import HuggingFaceLLM
from .litellm_client import LiteLLM
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .llama_client import LLaMAClient

def create_llm(provider: str, model: str, **kwargs):
    if 'api_key' in kwargs:
        print(f"API Key found in kwargs for {provider}.")
    if provider == "huggingface":
        return HuggingFaceLLM(model, **kwargs)
    elif provider == "litellm":
        return LiteLLM(model, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model, **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model, **kwargs)
    elif provider == "llama":
        return LLaMAClient(model, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graphfusionai.llm import create_llm

def main():
    messages = [{"role": "user", "content": "Tell me a joke."}]
    llama_llm = create_llm(provider="llama", api_key="", model="facebook/llama-7b")
    response = llama_llm.call(messages)
    print(f"LLaMA Response: {response}")

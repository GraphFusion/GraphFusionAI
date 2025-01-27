
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graphfusionai.llm import create_llm

llm = create_llm("huggingface", model="gpt2")
messages = [{"role": "user", "content": "Tell me a joke."}]
print(llm.call(messages))

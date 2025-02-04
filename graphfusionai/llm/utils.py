
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict

def trim_messages(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Trim messages to fit within max_tokens."""
    token_count = 0
    trimmed_messages = []
    for msg in reversed(messages):
        token_count += len(msg["content"].split())  
        if token_count > max_tokens:
            break
        trimmed_messages.append(msg)
    return list(reversed(trimmed_messages))
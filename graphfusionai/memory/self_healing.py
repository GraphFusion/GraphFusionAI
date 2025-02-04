import torch
from typing import Dict, Any, List

class SelfHealingMemory:
    """
    Handles automatic pruning of the memory store by removing outdated, irrelevant, 
    or low-confidence memories.
    """

    def __init__(self, threshold: float = 0.5, max_size: int = 1000):
        """
        Initializes the self-healing memory mechanism.

        Args:
            threshold (float): Confidence score threshold for removing low-relevance memories.
            max_size (int): Maximum allowed memory entries before pruning.
        """
        self.threshold = threshold
        self.max_size = max_size

    def prune_low_confidence(self, memory_store: Dict[str, Any]) -> None:
        """
        Removes memories with confidence scores below the threshold.

        Args:
            memory_store (dict): The stored memory knowledge.
        """
        keys_to_remove = [key for key, value in memory_store.items() if value.get("score", 1.0) < self.threshold]

        for key in keys_to_remove:
            del memory_store[key]

    def prune_oldest_entries(self, memory_store: Dict[str, Any]) -> None:
        """
        Ensures memory store size remains within the allowed limit by removing the oldest entries.

        Args:
            memory_store (dict): The stored memory knowledge.
        """
        if len(memory_store) > self.max_size:
            sorted_keys = sorted(memory_store.keys(), key=lambda k: memory_store[k].get("timestamp", 0))
            excess = len(memory_store) - self.max_size
            for key in sorted_keys[:excess]:
                del memory_store[key]

    def heal_memory(self, memory_store: Dict[str, Any]) -> None:
        """
        Performs full self-healing by pruning both low-confidence and outdated entries.

        Args:
            memory_store (dict): The stored memory knowledge.
        """
        self.prune_low_confidence(memory_store)
        self.prune_oldest_entries(memory_store)

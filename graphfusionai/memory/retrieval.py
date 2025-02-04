import torch
from typing import Dict, List, Any
from scipy.spatial.distance import cosine

class MemoryRetrieval:
    """
    Handles memory retrieval by performing similarity search on stored memory embeddings.
    """

    def search(self, memory_store: Dict[str, Any], query_vector: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant memories based on vector similarity.

        Args:
            memory_store (dict): The stored memory knowledge.
            query_vector (torch.Tensor): The vector representation of the query.
            top_k (int): Number of most relevant results to return.

        Returns:
            List[Dict[str, Any]]: Top matching memories sorted by relevance.
        """
        scores = []

        # Compute cosine similarity between query and stored memories
        for key, memory in memory_store.items():
            memory_vector = torch.tensor([float(x) for x in key.strip("()").split(",")])
            similarity = 1 - cosine(query_vector.tolist(), memory_vector.tolist())  # Cosine similarity
            scores.append((key, similarity))

        # Sort by highest similarity score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for key, score in scores[:top_k]:
            memory_data = memory_store[key]
            results.append({"memory_key": key, "data": memory_data["data"], "metadata": memory_data.get("metadata"), "score": score})

        return results

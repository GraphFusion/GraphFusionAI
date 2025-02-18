import torch
from typing import Dict, List, Any, Optional
from scipy.spatial.distance import cosine
from graphfusionai.memory.embeddings import EmbeddingModel

class MemoryRetrieval:
    """
    Handles memory retrieval by performing similarity search on stored memory embeddings.
    Provides efficient and flexible memory search capabilities.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize the memory retrieval system.

        Args:
            embedding_model: Model to generate embeddings. If None, creates default model.
        """
        self.embedding_model = embedding_model or EmbeddingModel()

    def search(self, 
              memory_store: Dict[str, Any], 
              query_vector: torch.Tensor, 
              top_k: int = 3,
              min_similarity: float = 0.0,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant memories based on vector similarity.

        Args:
            memory_store (dict): The stored memory knowledge.
            query_vector (torch.Tensor): The vector representation of the query.
            top_k (int): Number of most relevant results to return.
            min_similarity (float): Minimum similarity threshold for results.
            filter_metadata (dict): Optional metadata filters to apply.

        Returns:
            List[Dict[str, Any]]: Top matching memories sorted by relevance.
        """
        scores = []

        # Calculate similarities in batches for better performance
        memory_vectors = []
        memory_keys = []
        
        for key, memory in memory_store.items():
            # Skip if metadata filter doesn't match
            if filter_metadata and not self._matches_metadata(memory.get("metadata", {}), filter_metadata):
                continue
                
            memory_vector = memory.get("embedding")
            if memory_vector is None:
                memory_vector = torch.tensor([float(x) for x in key.strip("()").split(",")])
            
            memory_vectors.append(memory_vector)
            memory_keys.append(key)

        if memory_vectors:
            # Batch compute similarities
            memory_tensors = torch.stack(memory_vectors)
            similarities = 1 - torch.tensor([
                cosine(query_vector.tolist(), mv.tolist()) 
                for mv in memory_tensors
            ])

            # Filter by minimum similarity
            for key, similarity in zip(memory_keys, similarities):
                if similarity >= min_similarity:
                    scores.append((key, float(similarity)))

        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for key, score in scores[:top_k]:
            memory_data = memory_store[key]
            results.append({
                "memory_key": key,
                "data": memory_data["data"],
                "metadata": memory_data.get("metadata"),
                "score": score
            })

        return results

    def _matches_metadata(self, memory_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Helper method to check if memory metadata matches filter criteria."""
        return all(
            memory_metadata.get(key) == value 
            for key, value in filter_metadata.items()
        )

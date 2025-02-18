import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from typing import Dict, List, Any, Optional, Union
from graphfusionai.memory.dynamic_memory_cell import DynamicMemoryCell
from graphfusionai.memory.embeddings import EmbeddingModel
from graphfusionai.memory.retrieval import MemoryRetrieval

class MemoryManager:
    """
    Centralized memory controller for GraphFusionAI.
    Handles memory storage, retrieval, and updates with advanced memory management features.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initializes the memory system.

        Args:
            embedding_model: Model to generate embeddings. If None, creates default model.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.memory_cell = DynamicMemoryCell(self.embedding_model)
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.memory_tags: Dict[str, List[str]] = {}  # For organizing memories by tags
        self.memory_timestamps: Dict[str, float] = {}  # For tracking memory age
        self.importance_scores: Dict[str, float] = {}  # For prioritizing memories
        self.retrieval = MemoryRetrieval(embedding_model=self.embedding_model)

    def store_memory(self, 
                    data: Union[str, List[str]], 
                    tags: Optional[List[str]] = None,
                    importance: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """
        Stores knowledge as vector embeddings in memory with enhanced metadata.

        Args:
            data: Knowledge to be stored (single text or list of texts)
            tags: Optional list of tags to organize memories
            importance: Importance score (0.0 to 1.0) for memory prioritization
            metadata: Additional context for retrieval

        Returns:
            Unique memory key(s) (single key or list of keys)
        """
        if isinstance(data, str):
            data = [data]
            single_input = True
        else:
            single_input = False

        memory_keys = []
        for text in data:
            vector = self.embedding_model.encode(text)
            memory_key = str(hash(tuple(vector.tolist())))
            
            self.memory_store[memory_key] = {
                "data": text,
                "metadata": metadata or {},
                "embedding": vector
            }
            
            if tags:
                self.memory_tags[memory_key] = tags
                
            self.importance_scores[memory_key] = max(0.0, min(1.0, importance))
            self.memory_timestamps[memory_key] = torch.cuda.Event().record()
            
            self.memory_cell.add(text)
            memory_keys.append(memory_key)

        return memory_keys[0] if single_input else memory_keys

    def retrieve_memory(self, 
                       query: str,
                       top_k: int = 5,
                       min_similarity: float = 0.0,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant memories based on the query with filtering options.

        Args:
            query: The input query for retrieval
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold for results
            tags: Optional list of tags to filter memories

        Returns:
            List of relevant memory entries with similarity scores
        """
        results = self.memory_cell.query(query, top_k=top_k)
        
        results = [r for r in results if r["similarity"] >= min_similarity]
        
        # Filter by tags if provided
        if tags:
            filtered_results = []
            for result in results:
                memory_key = self._get_key_by_text(result["text"])
                if memory_key and any(tag in self.memory_tags.get(memory_key, []) for tag in tags):
                    filtered_results.append(result)
            results = filtered_results

        # Enhance results with metadata
        for result in results:
            memory_key = self._get_key_by_text(result["text"])
            if memory_key:
                result["metadata"] = self.memory_store[memory_key].get("metadata", {})
                result["importance"] = self.importance_scores.get(memory_key, 1.0)
                
        return results

    def update_memory(self, 
                     memory_key: str, 
                     new_data: Optional[str] = None,
                     new_tags: Optional[List[str]] = None,
                     new_importance: Optional[float] = None,
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Updates an existing memory entry with flexible field updates.

        Args:
            memory_key: The key of the memory to update
            new_data: Optional updated information
            new_tags: Optional updated tags
            new_importance: Optional updated importance score
            new_metadata: Optional updated metadata

        Returns:
            Whether the update was successful
        """
        if memory_key not in self.memory_store:
            return False
            
        if new_data:
            old_data = self.memory_store[memory_key]["data"]
            self.memory_cell.remove(old_data)  # Remove old data from cell
            self.memory_store[memory_key]["data"] = new_data
            self.memory_store[memory_key]["embedding"] = self.embedding_model.encode(new_data)
            self.memory_cell.add(new_data)  # Add new data to cell
            
        if new_tags is not None:
            self.memory_tags[memory_key] = new_tags
            
        if new_importance is not None:
            self.importance_scores[memory_key] = max(0.0, min(1.0, new_importance))
            
        if new_metadata is not None:
            self.memory_store[memory_key]["metadata"].update(new_metadata)
            
        return True

    def _get_key_by_text(self, text: str) -> Optional[str]:
        """Helper method to find memory key by stored text."""
        for key, value in self.memory_store.items():
            if value["data"] == text:
                return key
        return None

    def clear(self) -> None:
        """Clears all stored memories and related data."""
        self.memory_store.clear()
        self.memory_tags.clear()
        self.memory_timestamps.clear()
        self.importance_scores.clear()
        self.memory_cell.clear()

    def search_memories(self, query: str, top_k: int = 3, min_similarity: float = 0.0, 
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search memories using the query string.
        
        Args:
            query: The search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching memories with their similarity scores
        """
        query_vector = self.embedding_model.encode(query)
        return self.retrieval.search(
            self.memory_store,
            query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata
        )

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from typing import Dict, List, Any, Optional
from graphfusionai.memory.dynamic_memory_cell import DynamicMemoryCell
from graphfusionai.memory.retrieval import MemoryRetrieval
from graphfusionai.memory.embeddings import EmbeddingModel

class MemoryManager:
    """
    Centralized memory controller for GraphFusionAI.
    Handles memory storage, retrieval, and updates.
    """

    def __init__(self, input_dim: int = 256, memory_dim: int = 512, context_dim: int = 128):
        """
        Initializes the memory system with a dynamic memory cell.

        Args:
            input_dim (int): Input feature size.
            memory_dim (int): Memory storage size.
            context_dim (int): Context embedding size.
        """
        self.memory_cell = DynamicMemoryCell(input_dim, memory_dim, context_dim)
        self.memory_retrieval = MemoryRetrieval()
        self.embedding_model = EmbeddingModel()

        self.memory_store: Dict[str, Any] = {}

    def store_memory(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores knowledge as vector embeddings in memory.

        Args:
            data (str): Knowledge to be stored.
            metadata (dict, optional): Additional context for retrieval.

        Returns:
            str: Unique memory key (vector hash).
        """
        vector = self.embedding_model.encode(data)
        memory_key = str(hash(tuple(vector.tolist())))  
        self.memory_store[memory_key] = {"data": data, "metadata": metadata}
        return memory_key

    def retrieve_memory(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant memories based on the query.

        Args:
            query (str): The input query for retrieval.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: List of relevant memory entries.
        """
        query_vector = self.embedding_model.encode(query)
        return self.memory_retrieval.search(self.memory_store, query_vector, top_k)

    def update_memory(self, memory_key: str, new_data: str) -> bool:
        """
        Updates an existing memory entry.

        Args:
            memory_key (str): The key of the memory to update.
            new_data (str): The updated information.

        Returns:
            bool: Whether the update was successful.
        """
        if memory_key in self.memory_store:
            self.memory_store[memory_key]["data"] = new_data
            return True
        return False

    

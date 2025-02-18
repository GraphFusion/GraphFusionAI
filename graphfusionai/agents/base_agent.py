import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, List
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.memory.memory_manager import MemoryManager  
import torch

class BaseAgent:
    def __init__(self, 
                 name: str, 
                 graph_network: GraphNetwork, 
                 knowledge_graph: KnowledgeGraph, 
                 llm_provider: str,  
                 api_key: str,       
                 model: str,
                 memory_manager: MemoryManager  
                 ):
        """
        Initializes the BaseAgent with the necessary components: 
        graph network, knowledge graph, memory management, and LLM client.
        """
        self.name = name
        self.graph_network = graph_network
        self.knowledge_graph = knowledge_graph
        self.memory_manager = memory_manager  


    def process_input(self, input_data: Any) -> Dict[str, torch.Tensor]:
        """
        Abstract method to process input data. Must be implemented in derived classes.
        This could process raw data, query the memory, or integrate LLM capabilities.
        """
        raise NotImplementedError

    def update_graph(self, updates: List[Dict[str, Any]]) -> None:
        """
        Updates the knowledge graph based on new information. This method updates 
        the graph with new relationships or facts.
        """
        for update in updates:
            self.knowledge_graph.add_relation(
                update['from'], update['to'], update['relation'], update.get('features')
            )

    def decide(self, input_data: Any) -> Any:
        """
        Abstract method for decision-making. Must be implemented in derived classes.
        This method can access both the knowledge graph and memory to make decisions.
        """
        raise NotImplementedError

    def communicate(self, other_agent: "BaseAgent", message: Any) -> None:
        """
        Facilitates inter-agent communication. Agents can share insights or 
        requests for reasoning by utilizing their LLMs and the memory system.
        """
        raise NotImplementedError

    def use_llm_for_query(self, query: str) -> str:
        """
        Uses the LLM to process a query. This allows the agent to leverage 
        context-driven interactions for better decision-making.
        """
        response = self.llm_client.call([{"role": "user", "content": query}])
        return response

    def query_memory(self, query: str) -> Any:
        """
        Queries the memory manager for relevant context or information. 
        Memory retrieval helps the agent make more informed decisions.
        """
        return self.memory_manager.retrieve(query)

    def update_memory(self, new_data: Any) -> None:
        """
        Updates the memory with new context or insights. The memory manager handles 
        the storage and retrieval of information for decision-making.
        """
        self.memory_manager.store(new_data)

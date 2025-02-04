import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, List, Optional
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.core.memory_cell import DynamicMemoryCell
from graphfusionai.llm import create_llm  
import torch

class BaseAgent:
    def __init__(self, 
                 name: str, 
                 graph_network: GraphNetwork, 
                 knowledge_graph: KnowledgeGraph, 
                 llm_provider: str,  
                 api_key: str,       
                 model: str           
                 ):
        self.name = name
        self.graph_network = graph_network
        self.knowledge_graph = knowledge_graph
        self.memory_cell = DynamicMemoryCell(input_dim=256, memory_dim=512, context_dim=128)

        self.llm_client = create_llm(provider=llm_provider, api_key=api_key, model=model)

    def process_input(self, input_data: Any) -> Dict[str, torch.Tensor]:
        """
        Abstract method to process input data. Must be implemented in derived classes.
        """
        raise NotImplementedError

    def update_graph(self, updates: List[Dict[str, Any]]) -> None:
        """
        Update the knowledge graph based on new information.
        """
        for update in updates:
            self.knowledge_graph.add_relation(
                update['from'], update['to'], update['relation'], update.get('features')
            )

    def decide(self, input_data: Any) -> Any:
        """
        Abstract method for decision-making. Must be implemented in derived classes.
        This can utilize the LLM to enhance decision-making with context and natural language.
        """
        raise NotImplementedError

    def communicate(self, other_agent: "BaseAgent", message: Any) -> None:
        """
        Abstract method for inter-agent communication.
        Agents can share insights or requests for reasoning by utilizing their LLMs.
        """
        raise NotImplementedError

    def use_llm_for_query(self, query: str) -> str:
        """
        Use the LLM for processing the query. This enables context-driven interactions
        by leveraging the LLM's capabilities.
        """
        response = self.llm_client.call([{"role": "user", "content": query}])
        return response

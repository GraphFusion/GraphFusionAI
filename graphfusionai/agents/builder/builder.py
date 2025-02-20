import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Any, Dict, Type, Union
from graphfusionai.agents.base_agent import BaseAgent
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.memory.memory_manager import MemoryManager  
from .validators import validate_config
from graphfusionai.tools.registry import ToolRegistry
from graphfusionai.llm import create_llm
import logging

logger = logging.getLogger(__name__)

class AgentBuilder:
    """
    A builder for dynamically creating agents based on specifications, 
    including tools and memory.
    """

    def __init__(self, graph_network: GraphNetwork, knowledge_graph: KnowledgeGraph):
        """
        Initialize the AgentBuilder with core components.

        Args:
            graph_network (GraphNetwork): The graph network for agent interaction.
            knowledge_graph (KnowledgeGraph): The shared knowledge graph.
        """
        self.graph_network = graph_network
        self.knowledge_graph = knowledge_graph

    def _create_llm_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates an LLM instance based on the configuration.

        Args:
            config (Dict[str, Any]): LLM configuration.

        Returns:
            Dict[str, Any]: LLM parameters.
        """
        llm_provider = config.get("llm_provider")
        api_key = config.get("api_key")
        model = config.get("model")

        if not all([llm_provider, api_key, model]):
            logger.error("Missing required LLM configuration parameters")
            raise ValueError("Missing required LLM configuration parameters")

        return {
            "llm_provider": llm_provider,
            "api_key": api_key,
            "model": model
        }

    def _create_memory_manager(self, config: Dict[str, Any]) -> MemoryManager:
        """
        Creates a MemoryManager instance based on the configuration.

        Args:
            config (Dict[str, Any]): Memory configuration.

        Returns:
            MemoryManager: The initialized memory manager.
        """
        memory_config = config.get("memory", {})
        return MemoryManager(
            input_dim=memory_config.get("input_dim", 256),
            memory_dim=memory_config.get("memory_dim", 512),
            context_dim=memory_config.get("context_dim", 128)
        )

    def create_agent(
        self,
        agent_type: Type[BaseAgent],
        name: str,
        config: Dict[str, Any]
    ) -> Union[BaseAgent, None]:
        """
        Dynamically creates an agent of the specified type with memory and tools.

        Args:
            agent_type (Type[BaseAgent]): The class type for the agent.
            name (str): The agent's name.
            config (Dict[str, Any]): Configuration for the agent.

        Returns:
            BaseAgent: An initialized agent instance.

        Raises:
            ValueError: If the configuration is invalid or tools are missing.
        """
        validate_config(config)
        
        llm_params = self._create_llm_instance(config)
        memory_manager = self._create_memory_manager(config)

        agent = agent_type(
            name=name,
            graph_network=self.graph_network,
            knowledge_graph=self.knowledge_graph,
            **llm_params,
            memory_manager=memory_manager  
        )

        tools = config.get("tools", [])
        for tool_name in tools:
            tool = ToolRegistry.get_tool(tool_name)
            if not tool:
                logger.warning(f"Invalid tool name: '{tool_name}', skipping...")
                continue
            agent.add_tool(tool())

        return agent

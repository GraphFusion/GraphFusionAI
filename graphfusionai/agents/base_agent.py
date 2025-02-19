import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid
from abc import ABC, abstractmethod
from copy import copy as shallow_copy
from hashlib import md5
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, validator, root_validator
from pydantic.types import UUID4

from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.memory.memory_manager import MemoryManager  
from graphfusionai.tools.base import BaseTool
from graphfusionai.llm import LiteLLMClient

T = TypeVar("T", bound="BaseAgent")


class BaseAgent(ABC, BaseModel):
    """
    A modular base agent for GraphFusionAI.
    
    Attributes:
        id (UUID4): Unique identifier for the agent.
        role (str): Role of the agent.
        goal (str): Objective of the agent.
        backstory (str): Backstory of the agent.
        config (Optional[Dict[str, Any]]): Agent configuration.
        cache (bool): Whether to use caching.
        verbose (bool): Enable verbose logging.
        max_rpm (Optional[int]): Maximum requests per minute.
        allow_delegation (bool): Whether the agent can delegate tasks.
        tools (Optional[List[Any]]): Tools available to the agent.
        max_iter (int): Maximum iterations for task execution.
        llm (Optional[LiteLLMClient]): Language model client.
        knowledge_graph (Optional[KnowledgeGraph]): The knowledge graph instance.
        memory_manager (Optional[MemoryManager]): Memory management instance.
    """

    __hash__ = object.__hash__

    _graph: GraphNetwork = PrivateAttr(default_factory=GraphNetwork)
    _memory_manager: MemoryManager = PrivateAttr(default_factory=MemoryManager)
    _llm_client: LiteLLMClient = PrivateAttr(default_factory=LiteLLMClient)
    _original_role: Optional[str] = PrivateAttr(default=None)
    _original_goal: Optional[str] = PrivateAttr(default=None)
    _original_backstory: Optional[str] = PrivateAttr(default=None)

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(..., description="Role of the agent")
    goal: str = Field(..., description="Objective of the agent")
    backstory: str = Field(..., description="Backstory of the agent")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Agent configuration", exclude=True)
    cache: bool = Field(default=True, description="Enable caching for tool usage")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    max_rpm: Optional[int] = Field(default=None, description="Maximum requests per minute")
    allow_delegation: bool = Field(default=False, description="Allow task delegation among agents")
    tools: Optional[List[Any]] = Field(default_factory=list, description="Tools available to the agent")
    max_iter: int = Field(default=25, description="Maximum iterations for task execution")
    llm: Optional[LiteLLMClient] = Field(default=None, description="Language model client")
    knowledge_graph: Optional[KnowledgeGraph] = Field(default=None, description="Knowledge graph instance")
    memory_manager: Optional[MemoryManager] = Field(default=None, description="Memory manager instance")

    @root_validator(pre=True)
    def process_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Process and validate configuration settings if needed.
        return values

    @validator("tools", pre=True, always=True)
    def validate_tools(cls, tools: List[Any]) -> List[BaseTool]:
        processed_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed_tools.append(tool)
            elif hasattr(tool, "name") and hasattr(tool, "func") and hasattr(tool, "description"):
                processed_tools.append(BaseTool())
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}. Each tool must be an instance of BaseTool or have the required attributes.")
        return processed_tools

    @root_validator
    def validate_required_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for field_name in ["role", "goal", "backstory"]:
            if not values.get(field_name):
                raise ValueError(f"'{field_name}' must be provided.")
        return values

    @property
    def key(self) -> str:
        source = [
            self._original_role or self.role,
            self._original_goal or self.goal,
            self._original_backstory or self.backstory,
        ]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

    @abstractmethod
    def execute_task(self, task: Any, context: Optional[str] = None, tools: Optional[List[BaseTool]] = None) -> str:
        """
        Execute a given task with optional context and tools.
        """
        pass

    @abstractmethod
    def create_agent_executor(self, tools: Optional[List[BaseTool]] = None) -> None:
        """
        Create and configure the agent executor.
        """
        pass

    @abstractmethod
    def _parse_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """
        Parse and validate the provided tools.
        """
        pass

    @abstractmethod
    def get_delegation_tools(self, agents: List["BaseAgent"]) -> List[BaseTool]:
        """
        Return the tools to be used for delegating tasks to other agents.
        """
        pass

    @abstractmethod
    def get_output_converter(
        self, llm: Any, text: str, model: Optional[type[BaseModel]], instructions: str
    ) -> Any:
        """
        Get a converter to transform LLM outputs into structured data.
        """
        pass

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Interpolate external inputs into the agent's role, goal, and backstory.
        """
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        self.role = self._original_role.format(**inputs)
        self.goal = self._original_goal.format(**inputs)
        self.backstory = self._original_backstory.format(**inputs)

    def copy(self: T) -> T:
        """
        Create a shallow copy of the agent, excluding runtime-specific attributes.
        """
        exclude = {"id", "_graph", "_memory_manager", "_llm_client"}
        copied_data = self.dict(exclude=exclude)
        copied_agent = type(self)(**copied_data)
        copied_agent._graph = self._graph  
        copied_agent._memory_manager = self._memory_manager
        copied_agent._llm_client = self._llm_client
        return copied_agent

    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """
        Set the memory manager for the agent.
        """
        self.memory_manager = memory_manager
        self._memory_manager = memory_manager

    def set_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        """
        Set the knowledge graph instance for the agent.
        """
        self.knowledge_graph = knowledge_graph

    def set_llm_client(self, llm_client: LiteLLMClient) -> None:
        """
        Set the language model client for the agent.
        """
        self.llm = llm_client
        self._llm_client = llm_client


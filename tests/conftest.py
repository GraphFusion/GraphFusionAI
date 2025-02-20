"""
Shared test fixtures and utilities for GraphFusionAI tests.
"""
import pytest
from typing import Dict, Any
from graphfusionai import GraphNetwork
from graphfusionai.agents import BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.tools.base import BaseTool

class MockTool(BaseTool):
    def __init__(self, name: str = "mock_tool"):
        super().__init__(name=name)
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        return {"status": "success", "result": "mock result"}

class TestAgent(BaseAgent):
    def execute_task(self, task: Dict[str, Any], context=None, tools=None) -> str:
        return "test execution"

    def create_agent_executor(self, tools=None) -> Any:
        return self

    def _parse_tools(self, tools: list) -> list:
        return tools

    def get_delegation_tools(self, agents: list) -> list:
        return []

    def get_output_converter(self, llm, text, model, instructions):
        return text

@pytest.fixture
def mock_tool():
    """Fixture that provides a mock tool for testing."""
    return MockTool()

@pytest.fixture
def test_agent():
    """Fixture that provides a test agent instance."""
    return TestAgent(
        role="test_role",
        goal="test_goal",
        backstory="test_backstory"
    )

@pytest.fixture
def graph_network():
    """Fixture that provides a graph network instance."""
    return GraphNetwork(feature_dim=64, hidden_dim=32)

@pytest.fixture
def memory_manager():
    """Fixture that provides a memory manager instance."""
    return MemoryManager()

@pytest.fixture
def mock_llm_client():
    """Fixture that provides a mock LLM client."""
    class MockLLMClient(LiteLLMClient):
        def generate(self, prompt: str, **kwargs) -> str:
            return "mock response"
        
        def embed(self, text: str) -> list:
            return [0.0] * 64
    
    return MockLLMClient()

@pytest.fixture
def test_graph_data():
    """Fixture that provides test graph data."""
    return {
        "nodes": [
            {"id": "1", "type": "concept", "features": {"name": "AI"}},
            {"id": "2", "type": "concept", "features": {"name": "ML"}},
            {"id": "3", "type": "concept", "features": {"name": "DL"}}
        ],
        "edges": [
            {"from": "1", "to": "2", "relation": "includes"},
            {"from": "2", "to": "3", "relation": "includes"}
        ]
    }

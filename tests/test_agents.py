"""
Tests for the agent system in GraphFusionAI.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from graphfusionai.agents import BaseAgent
from graphfusionai.tools.base import BaseTool

def test_agent_initialization(test_agent):
    """Test basic agent initialization."""
    assert test_agent.role == "test_role"
    assert test_agent.goal == "test_goal"
    assert test_agent.backstory == "test_backstory"
    assert isinstance(test_agent, BaseAgent)

def test_agent_key_generation(test_agent):
    """Test that agent key is generated correctly."""
    key = test_agent.key
    assert isinstance(key, str)
    assert len(key) > 0

def test_agent_copy(test_agent, graph_network, memory_manager, mock_llm_client):
    """Test agent copy functionality."""
    # Setup agent with components
    test_agent._graph = graph_network
    test_agent.set_memory_manager(memory_manager)
    test_agent.set_llm_client(mock_llm_client)
    
    # Create copy
    copied_agent = test_agent.copy()
    
    # Verify copy
    assert copied_agent is not test_agent
    assert copied_agent.role == test_agent.role
    assert copied_agent.goal == test_agent.goal
    assert copied_agent.backstory == test_agent.backstory
    assert copied_agent._graph is test_agent._graph
    assert copied_agent._memory_manager is test_agent._memory_manager
    assert copied_agent._llm_client is test_agent._llm_client

def test_agent_tool_validation(test_agent, mock_tool):
    """Test tool validation."""
    tools = [mock_tool]
    validated_tools = test_agent.validate_tools(tools)
    assert len(validated_tools) == 1
    assert isinstance(validated_tools[0], BaseTool)

def test_agent_input_interpolation(test_agent):
    """Test input interpolation."""
    inputs = {
        "name": "TestBot",
        "task": "testing",
        "domain": "AI"
    }
    
    test_agent.role = "I am {name}"
    test_agent.goal = "I do {task}"
    test_agent.backstory = "I work in {domain}"
    
    test_agent.interpolate_inputs(inputs)
    
    assert test_agent.role == "I am TestBot"
    assert test_agent.goal == "I do testing"
    assert test_agent.backstory == "I work in AI"

def test_agent_memory_management(test_agent, memory_manager):
    """Test memory manager integration."""
    test_agent.set_memory_manager(memory_manager)
    assert test_agent.memory_manager is memory_manager
    assert test_agent._memory_manager is memory_manager

    with pytest.raises(ValueError):
        test_agent.set_memory_manager(None)

def test_agent_llm_integration(test_agent, mock_llm_client):
    """Test LLM client integration."""
    test_agent.set_llm_client(mock_llm_client)
    assert test_agent.llm is mock_llm_client
    assert test_agent._llm_client is mock_llm_client

    with pytest.raises(ValueError):
        test_agent.set_llm_client(None)

def test_agent_task_execution(test_agent, mock_tool):
    """Test task execution."""
    task = {"type": "test", "content": "test task"}
    tools = [mock_tool]
    
    result = test_agent.execute_task(task, tools=tools)
    assert isinstance(result, str)
    assert result == "test execution"

@pytest.mark.asyncio
async def test_agent_async_operations(test_agent, mock_tool):
    """Test async operations."""
    with patch.object(test_agent, 'execute_task') as mock_execute:
        mock_execute.return_value = "async result"
        
        task = {"type": "test", "content": "async test"}
        tools = [mock_tool]
        
        result = await test_agent.execute_task_async(task, tools=tools)
        assert result == "async result"
        mock_execute.assert_called_once_with(task, None, tools)

def test_agent_error_handling(test_agent):
    """Test error handling in agent operations."""
    # Test invalid input interpolation
    with pytest.raises(KeyError):
        test_agent.interpolate_inputs({"invalid": "input"})
    
    # Test invalid tool validation
    with pytest.raises(ValueError):
        test_agent.validate_tools([{"invalid": "tool"}])

def test_agent_cleanup(test_agent, memory_manager, mock_llm_client):
    """Test agent cleanup."""
    test_agent.set_memory_manager(memory_manager)
    test_agent.set_llm_client(mock_llm_client)
    
    # Mock cleanup methods
    memory_manager.cleanup = Mock()
    mock_llm_client.cleanup = Mock()
    
    test_agent.cleanup()
    
    # Verify cleanup calls
    memory_manager.cleanup.assert_called_once()
    mock_llm_client.cleanup.assert_called_once()
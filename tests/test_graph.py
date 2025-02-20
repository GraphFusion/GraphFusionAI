"""
Tests for the graph network system in GraphFusionAI.
"""
import pytest
import torch
import networkx as nx
from typing import Dict, Any

from graphfusionai.core.graph import GraphNetwork, GraphNode

def test_graph_node_initialization():
    """Test graph node initialization."""
    node = GraphNode(
        node_id="test_node",
        features={"name": "Test", "value": 1},
        node_type="test_type"
    )
    
    assert node.node_id == "test_node"
    assert node.features == {"name": "Test", "value": 1}
    assert node.node_type == "test_type"
    assert node.neighbors == {"in": [], "out": []}

def test_graph_node_neighbors():
    """Test graph node neighbor management."""
    node1 = GraphNode("node1")
    node2 = GraphNode("node2")
    
    node1.add_neighbor(node2, "out")
    node2.add_neighbor(node1, "in")
    
    assert node2 in node1.neighbors["out"]
    assert node1 in node2.neighbors["in"]
    
    # Test duplicate prevention
    node1.add_neighbor(node2, "out")
    assert len(node1.neighbors["out"]) == 1

def test_graph_network_initialization(graph_network):
    """Test graph network initialization."""
    assert isinstance(graph_network.graph, nx.DiGraph)
    assert graph_network.feature_dim == 64
    assert graph_network.hidden_dim == 32
    assert isinstance(graph_network.node_update, torch.nn.GRUCell)
    assert isinstance(graph_network.message_net, torch.nn.Sequential)
    assert isinstance(graph_network.attention, torch.nn.MultiheadAttention)

def test_graph_network_node_management(graph_network):
    """Test node addition and management in graph network."""
    node = graph_network.add_node(
        node_id="test",
        features={"value": 1.0},
        node_type="test_type"
    )
    
    assert isinstance(node, GraphNode)
    assert node.node_id == "test"
    assert node.features == {"value": 1.0}
    assert node.node_type == "test_type"
    assert "test" in graph_network.nodes
    assert "test" in graph_network.graph.nodes

def test_graph_network_edge_operations(graph_network):
    """Test edge operations in graph network."""
    # Add nodes
    node1 = graph_network.add_node("node1")
    node2 = graph_network.add_node("node2")
    
    # Add edge
    graph_network.graph.add_edge("node1", "node2", weight=1.0)
    
    # Verify edge
    assert graph_network.graph.has_edge("node1", "node2")
    assert graph_network.graph.edges[("node1", "node2")]["weight"] == 1.0

def test_graph_network_state_management(graph_network):
    """Test node state management."""
    node = graph_network.add_node("test")
    assert "test" in graph_network.node_states
    assert graph_network.node_states["test"].shape == (graph_network.hidden_dim,)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graph_network_gpu_support(graph_network):
    """Test GPU support for graph network."""
    graph_network.cuda()
    
    # Add node and verify tensor device
    node = graph_network.add_node("test")
    assert graph_network.node_states["test"].is_cuda

def test_graph_network_batch_processing(graph_network):
    """Test batch processing of nodes."""
    # Create a batch of nodes
    nodes = [
        graph_network.add_node(f"node{i}") 
        for i in range(3)
    ]
    
    # Create feature tensors
    features = torch.randn(3, graph_network.feature_dim)
    
    # Process batch
    output = graph_network(features)
    assert output.shape == (3, graph_network.hidden_dim)

def test_graph_network_message_passing(graph_network):
    """Test message passing between nodes."""
    # Create connected nodes
    node1 = graph_network.add_node("node1")
    node2 = graph_network.add_node("node2")
    graph_network.graph.add_edge("node1", "node2")
    
    # Create feature tensors
    features = torch.randn(2, graph_network.feature_dim)
    
    # Process message passing
    output = graph_network(features)
    assert output.shape == (2, graph_network.hidden_dim)

def test_graph_network_attention(graph_network):
    """Test attention mechanism."""
    # Create nodes
    nodes = [graph_network.add_node(f"node{i}") for i in range(3)]
    
    # Create complete graph
    for i in range(3):
        for j in range(3):
            if i != j:
                graph_network.graph.add_edge(f"node{i}", f"node{j}")
    
    # Create feature tensors
    features = torch.randn(3, graph_network.feature_dim)
    
    # Process with attention
    output = graph_network(features)
    assert output.shape == (3, graph_network.hidden_dim)

def test_graph_network_serialization(graph_network, tmp_path):
    """Test graph network serialization."""
    # Add some nodes and edges
    node1 = graph_network.add_node("node1")
    node2 = graph_network.add_node("node2")
    graph_network.graph.add_edge("node1", "node2")
    
    # Save network
    save_path = tmp_path / "graph_network.pt"
    torch.save(graph_network.state_dict(), save_path)
    
    # Load network
    new_network = GraphNetwork(graph_network.feature_dim, graph_network.hidden_dim)
    new_network.load_state_dict(torch.load(save_path))
    
    # Verify parameters
    for p1, p2 in zip(graph_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)
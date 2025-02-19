import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any, Dict, List, Optional
import networkx as nx
import torch

class KnowledgeGraph:
    """
    A knowledge graph implementation that stores entities and their relationships.
    Supports features on nodes and edges, and provides graph querying capabilities.
    """

    def __init__(self):
        """
        Initialize an empty knowledge graph using NetworkX.
        """
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.edge_features = {}

    def add_node(self, node_id: str, features: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a node to the knowledge graph with optional features.

        Args:
            node_id: Unique identifier for the node
            features: Optional dictionary of node features
        """
        self.graph.add_node(node_id)
        if features:
            self.node_features[node_id] = features

    def add_relation(self, from_node: str, to_node: str, relation: str, 
                    features: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relation (edge) between two nodes with optional features.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            relation: Type of relationship
            features: Optional dictionary of edge features
        """
        self.add_node(from_node)
        self.add_node(to_node)

        self.graph.add_edge(from_node, to_node, relation=relation)
        if features:
            self.edge_features[(from_node, to_node)] = features

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get all neighboring nodes for a given node.

        Args:
            node_id: ID of the node to get neighbors for

        Returns:
            List of neighboring node IDs
        """
        return list(self.graph.successors(node_id))

    def get_node_features(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Dictionary of node features if they exist, None otherwise
        """
        return self.node_features.get(node_id)

    def get_edge_features(self, from_node: str, to_node: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a specific edge.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            Dictionary of edge features if they exist, None otherwise
        """
        return self.edge_features.get((from_node, to_node))

    def get_relations(self, from_node: str, to_node: str) -> List[str]:
        """
        Get all relations between two nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            List of relation types between the nodes
        """
        return [self.graph[from_node][to_node]['relation']] if self.graph.has_edge(from_node, to_node) else []

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the graph structure to a tensor representation.

        Returns:
            Tensor representation of the graph
        """
        return torch.tensor(nx.to_numpy_array(self.graph), dtype=torch.float32)

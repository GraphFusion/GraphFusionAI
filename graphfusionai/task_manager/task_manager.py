import os
import sys
import logging
import asyncio
import torch
import networkx as nx
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from task_executor import TaskExecutor
from agent_manager import AgentManager
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph

class TaskManager:
    """Neural graph-powered task manager for orchestrating AI agent workflows."""
    
    def __init__(self, feature_dim=64, hidden_dim=128, dynamic_agents=True):
        """
        Initializes the TaskManager with integrated neural graph components.

        Args:
            feature_dim (int): Dimension of input features for the graph network
            hidden_dim (int): Hidden dimension for the graph network
            dynamic_agents (bool): If True, agents are dynamically assigned
        """
        # Initialize neural graph network for task dependencies
        self.graph_network = GraphNetwork(feature_dim, hidden_dim)
        
        # Initialize knowledge graph for task metadata
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize other components
        self.agent_manager = AgentManager(dynamic=dynamic_agents)
        self.executor = TaskExecutor()

    def add_task(self, task_name, dependencies=None, priority=0, features=None):
        """
        Adds a task to the system with neural graph integration.

        Args:
            task_name (str): The name of the task
            dependencies (list, optional): List of dependent tasks
            priority (int, optional): Task priority (lower value = higher priority)
            features (dict, optional): Task features for the graph network
        """
        # Add task to neural graph network
        if features is None:
            features = {'priority': priority}
        self.graph_network.add_node(task_name, features=features)
        
        # Add task to knowledge graph with metadata
        self.knowledge_graph.add_node(task_name, features={
            'priority': priority,
            'status': 'pending',
            'dependencies': dependencies or []
        })
        
        # Add dependencies
        if dependencies:
            for dep in dependencies:
                self.graph_network.add_edge(dep, task_name, edge_type='depends_on')
                self.knowledge_graph.add_relation(
                    dep, task_name, 'depends_on',
                    features={'weight': priority}
                )

    def run_task(self, task_name):
        """
        Runs a single task using neural graph-based prioritization.

        Args:
            task_name (str): The task to execute
        """
        # Get task state from graph network
        task_state = self.graph_network.node_states.get(task_name)
        
        # Get available agent considering task state
        agent = self.agent_manager.get_available_agent(task_state=task_state)
        
        if agent:
            # Update knowledge graph
            self.knowledge_graph.add_relation(
                agent.name, task_name, 'executes',
                features={'timestamp': time.time()}
            )
            
            # Execute task
            self.executor.execute(agent, task_name)
        else:
            print(f"No available agents for '{task_name}', retrying later...")

    def execute_all(self):
        """Executes all tasks using neural message passing for optimal ordering."""
        # Perform message passing to update task states
        task_states = self.graph_network.message_passing()
        
        # Get execution order based on graph network states
        execution_order = sorted(
            task_states.items(),
            key=lambda x: torch.sum(x[1]).item()
        )
        
        for task_name, _ in execution_order:
            self.run_task(task_name)

    def visualize(self):
        """Visualizes both the neural graph and knowledge graph."""
        # Visualize neural graph network states
        states = self.graph_network.node_states
        nx.draw(
            self.graph_network.graph,
            node_color=[torch.sum(states[n]).item() for n in self.graph_network.graph.nodes()],
            with_labels=True
        )
        
        # Visualize knowledge graph
        nx.draw(self.knowledge_graph.graph, with_labels=True)

# Example Usage
if __name__ == "__main__":
    # Initialize Task Manager with neural graph dimensions
    task_manager = TaskManager(feature_dim=64, hidden_dim=128)

    # Add tasks with dependencies and features
    task_manager.add_task(
        "Data Processing",
        dependencies=["Data Collection"],
        priority=1,
        features={'type': 'processing', 'resources': ['cpu', 'memory']}
    )

    # Execute all tasks - will use neural scheduling
    task_manager.execute_all()

    # Visualize task dependencies
    task_manager.visualize()

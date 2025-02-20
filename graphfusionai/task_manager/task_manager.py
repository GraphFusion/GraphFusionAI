import os
import sys
import logging
import asyncio
import torch
import networkx as nx
import time
from collections import defaultdict
import numpy as np

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
        
        # Enhanced task tracking
        self.task_history = {}
        self.task_patterns = defaultdict(list)
        self.task_analytics = defaultdict(lambda: {
            'success_rate': 0,
            'avg_duration': 0,
            'complexity_score': 0,
            'resource_usage': []
        })

    def add_task(self, task_name, dependencies=None, priority=0, features=None,
                required_capabilities=None, required_roles=None, complexity=1.0):
        """
        Adds a task to the system with enhanced metadata and requirements.

        Args:
            task_name (str): The name of the task
            dependencies (list, optional): List of dependent tasks
            priority (int, optional): Task priority (lower value = higher priority)
            features (dict, optional): Task features for the graph network
            required_capabilities (list): Required agent capabilities
            required_roles (list): Required agent roles
            complexity (float): Task complexity score (1.0 = normal)
        """
        # Add task to neural graph network
        if features is None:
            features = {
                'priority': priority,
                'complexity': complexity,
                'capabilities': required_capabilities or [],
                'roles': required_roles or []
            }
        self.graph_network.add_node(task_name, features=features)
        
        # Add task to knowledge graph with enhanced metadata
        self.knowledge_graph.add_node(task_name, features={
            'priority': priority,
            'status': 'pending',
            'dependencies': dependencies or [],
            'capabilities': required_capabilities or [],
            'roles': required_roles or [],
            'complexity': complexity,
            'creation_time': time.time()
        })
        
        # Add dependencies with enhanced relationship features
        if dependencies:
            for dep in dependencies:
                self.graph_network.add_edge(dep, task_name, edge_type='depends_on')
                self.knowledge_graph.add_relation(
                    dep, task_name, 'depends_on',
                    features={
                        'weight': priority,
                        'complexity_factor': complexity
                    }
                )
        
        # Initialize task history
        self.task_history[task_name] = {
            'status': 'pending',
            'attempts': 0,
            'assigned_agents': [],
            'start_time': None,
            'completion_time': None
        }

    def run_task(self, task_name):
        """
        Runs a single task using enhanced agent selection and monitoring.

        Args:
            task_name (str): The task to execute
        """
        # Get task state and requirements
        task_state = self.graph_network.node_states.get(task_name)
        task_features = self.knowledge_graph.get_node_features(task_name)
        
        # Form optimal team based on task requirements
        team = self.agent_manager.form_team({
            'capabilities': task_features['capabilities'],
            'roles': task_features['roles']
        })
        
        if team:
            # Update task history
            self.task_history[task_name].update({
                'status': 'running',
                'attempts': self.task_history[task_name]['attempts'] + 1,
                'assigned_agents': [agent.name for agent in team],
                'start_time': time.time()
            })
            
            # Update knowledge graph
            for agent in team:
                self.knowledge_graph.add_relation(
                    agent.name, task_name, 'executes',
                    features={
                        'timestamp': time.time(),
                        'role': agent.role
                    }
                )
            
            # Execute task with team
            success = self.executor.execute_with_team(team, task_name)
            
            # Update task analytics
            self._update_task_analytics(task_name, success)
            
            # Learn task patterns
            self._learn_task_patterns(task_name, team, success)
        else:
            print(f"No suitable team available for '{task_name}', retrying later...")

    def _update_task_analytics(self, task_name, success):
        """Updates analytics for a completed task."""
        history = self.task_history[task_name]
        analytics = self.task_analytics[task_name]
        
        # Calculate duration
        duration = time.time() - history['start_time']
        
        # Update success rate
        total_attempts = history['attempts']
        analytics['success_rate'] = ((analytics['success_rate'] * (total_attempts - 1) +
                                    (1 if success else 0)) / total_attempts)
        
        # Update average duration
        analytics['avg_duration'] = ((analytics['avg_duration'] * (total_attempts - 1) +
                                    duration) / total_attempts)
        
        # Update resource usage
        analytics['resource_usage'].append({
            'team_size': len(history['assigned_agents']),
            'duration': duration,
            'success': success
        })

    def _learn_task_patterns(self, task_name, team, success):
        """Learns patterns from task execution for future optimization."""
        if success:
            pattern = {
                'team_composition': [agent.role for agent in team],
                'team_size': len(team),
                'capabilities': set().union(*[agent.capabilities for agent in team]),
                'task_features': self.graph_network.node_states[task_name].tolist()
            }
            self.task_patterns[task_name].append(pattern)

    def get_task_recommendations(self, task_features):
        """
        Gets team and execution recommendations based on learned patterns.
        
        Args:
            task_features (dict): Features of the new task
            
        Returns:
            dict: Recommendations for team composition and execution strategy
        """
        recommendations = {
            'team_size': 0,
            'recommended_roles': set(),
            'required_capabilities': set(),
            'estimated_duration': 0
        }
        
        # Find similar tasks based on features
        similar_tasks = []
        task_vector = torch.tensor(list(task_features.values()))
        
        for task_name, patterns in self.task_patterns.items():
            for pattern in patterns:
                similarity = torch.cosine_similarity(
                    task_vector,
                    torch.tensor(pattern['task_features']),
                    dim=0
                )
                if similarity > 0.8:  # Similarity threshold
                    similar_tasks.append(pattern)
        
        if similar_tasks:
            # Aggregate recommendations
            recommendations['team_size'] = int(np.mean([p['team_size'] 
                                                      for p in similar_tasks]))
            recommendations['recommended_roles'] = set().union(
                *[set(p['team_composition']) for p in similar_tasks]
            )
            recommendations['required_capabilities'] = set().union(
                *[p['capabilities'] for p in similar_tasks]
            )
            
            # Estimate duration from analytics
            similar_task_analytics = [self.task_analytics[task_name] 
                                    for task_name in self.task_patterns.keys()
                                    if self.task_patterns[task_name]]
            if similar_task_analytics:
                recommendations['estimated_duration'] = np.mean(
                    [analytics['avg_duration'] for analytics in similar_task_analytics]
                )
        
        return recommendations

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
        features={'type': 'processing', 'resources': ['cpu', 'memory']},
        required_capabilities=['data_processing'],
        required_roles=['processor'],
        complexity=1.2
    )

    # Execute all tasks - will use neural scheduling
    task_manager.execute_all()

    # Visualize task dependencies
    task_manager.visualize()

    # Get task recommendations
    task_features = {
        'type': 'processing',
        'resources': ['cpu', 'memory'],
        'priority': 1
    }
    recommendations = task_manager.get_task_recommendations(task_features)
    print(recommendations)

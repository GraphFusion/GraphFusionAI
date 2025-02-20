import threading
import time
from queue import Queue
from collections import defaultdict
import torch
import networkx as nx

class Agent:
    """Represents an AI Agent that executes tasks."""
    
    def __init__(self, name, capabilities=None, embedding_dim=128):
        """
        Initializes an Agent.

        Args:
            name (str): The agent's name
            capabilities (list): List of skills the agent can handle
            embedding_dim (int): Dimension of the agent's state embedding
        """
        self.name = name
        self.capabilities = capabilities or []
        self.status = "Idle"
        self.current_task = None
        self.state_embedding = torch.zeros(embedding_dim)

    def update_state(self, task_state):
        """Updates agent's state based on current task."""
        if task_state is not None:
            self.state_embedding = task_state.clone()

    def assign_task(self, task, manager):
        """Assigns a task to the agent."""
        if self.status == "Idle":
            self.current_task = task
            self.status = "Busy"
            print(f"🚀 {self.name} started task: {task}")
            time.sleep(2)  
            self.complete_task(manager)

    def complete_task(self, manager):
        """Marks task as completed and notifies manager."""
        print(f"✅ {self.name} completed task: {self.current_task.name}")
        manager.mark_task_completed(self.current_task)
        self.current_task = None
        self.status = "Idle"
        self.state_embedding.zero_()


class Task:
    """Represents a task with dependencies and required capabilities."""
    
    def __init__(self, name, priority=1, duration=2, dependencies=None, required_capabilities=None, required_roles=None, task_state=None):
        """
        Initializes a task.

        Args:
            name (str): Task name.
            priority (int): Task priority (higher value = higher priority).
            duration (int): Time (seconds) the task takes to complete.
            dependencies (list): List of dependent task names.
            required_capabilities (list): List of capabilities needed.
            required_roles (list): List of roles needed.
            task_state (torch.Tensor): Neural state of the task
        """
        self.name = name
        self.priority = priority
        self.duration = duration
        self.dependencies = dependencies or []
        self.required_capabilities = required_capabilities or []
        self.required_roles = required_roles or []
        self.completed = False
        self.task_state = task_state
        self.assigned_team = None

    def is_ready(self, completed_tasks):
        """Checks if all dependencies are met before execution."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_completed(self):
        """Marks the task as completed."""
        self.completed = True

    def __str__(self):
        """String representation for user-friendly printing."""
        return f"{self.name} (Priority: {self.priority}, Duration: {self.duration}s)"

    def __repr__(self):
        """Debugging-friendly representation."""
        return f"Task(name={self.name}, priority={self.priority}, duration={self.duration}, dependencies={self.dependencies})"


class AgentManager:
    """Manages multiple AI agents and task assignments using neural graph states."""
    
    def __init__(self, dynamic=True, embedding_dim=128):
        self.agents = []
        self.task_queue = Queue()
        self.completed_tasks = set()
        self.lock = threading.Lock()
        self.dynamic = dynamic
        self.embedding_dim = embedding_dim
        
        # Enhanced agent coordination
        self.agent_states = {}  # Track agent neural states
        self.agent_roles = defaultdict(list)  # Track agent specializations
        self.collaboration_graph = nx.Graph()  # Track agent collaborations
        self.performance_metrics = defaultdict(lambda: {
            'tasks_completed': 0,
            'avg_completion_time': 0,
            'success_rate': 0,
            'collaboration_score': 0
        })

    def register_agent(self, name, capabilities=None, role=None):
        """
        Registers a new agent with enhanced role and capability tracking.
        
        Args:
            name (str): Agent name
            capabilities (list): Agent capabilities
            role (str): Primary role/specialization
        """
        agent = Agent(name, capabilities, self.embedding_dim)
        self.agents.append(agent)
        self.agent_states[name] = torch.zeros(self.embedding_dim)
        
        if role:
            self.agent_roles[role].append(name)
        
        # Add to collaboration graph
        self.collaboration_graph.add_node(name, role=role)
        return agent

    def form_team(self, task_requirements):
        """
        Forms an optimal team of agents for a complex task.
        
        Args:
            task_requirements (dict): Required capabilities and roles
        
        Returns:
            list: Selected team of agents
        """
        team = []
        required_capabilities = task_requirements.get('capabilities', [])
        required_roles = task_requirements.get('roles', [])
        
        # Score agents based on capabilities and roles
        agent_scores = {}
        for agent in self.agents:
            if agent.status == "Idle":
                # Calculate capability match
                capability_score = sum(1 for cap in required_capabilities 
                                    if cap in agent.capabilities)
                
                # Calculate role match
                role_score = sum(1 for role in required_roles 
                               if agent.name in self.agent_roles[role])
                
                # Consider past performance
                performance = self.performance_metrics[agent.name]
                performance_score = (performance['success_rate'] * 0.4 + 
                                  performance['collaboration_score'] * 0.6)
                
                # Calculate final score
                agent_scores[agent] = (capability_score * 0.4 + 
                                     role_score * 0.3 + 
                                     performance_score * 0.3)
        
        # Select top scoring agents
        sorted_agents = sorted(agent_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        team = [agent for agent, _ in sorted_agents[:len(required_roles)]]
        return team

    def assign_task(self, task, team=None):
        """
        Assigns a task to a team of agents.
        
        Args:
            task (Task): Task to assign
            team (list): Optional pre-selected team
        """
        if not team:
            team = self.form_team({
                'capabilities': task.required_capabilities,
                'roles': task.required_roles
            })
        
        if team:
            # Update collaboration graph
            for i, agent1 in enumerate(team):
                for agent2 in team[i+1:]:
                    if self.collaboration_graph.has_edge(agent1.name, agent2.name):
                        self.collaboration_graph[agent1.name][agent2.name]['weight'] += 1
                    else:
                        self.collaboration_graph.add_edge(agent1.name, agent2.name, weight=1)
            
            # Assign task to team
            task.assigned_team = team
            for agent in team:
                agent.assign_task(task, self)

    def mark_task_completed(self, task):
        """Marks a task as completed."""
        with self.lock:
            self.completed_tasks.add(task.name)
            print(f"🎯 Task '{task.name}' marked as completed.")

    def update_performance_metrics(self, agent_name, task_result):
        """Updates agent performance metrics after task completion."""
        metrics = self.performance_metrics[agent_name]
        metrics['tasks_completed'] += 1
        
        # Update success rate
        if task_result['success']:
            new_success = ((metrics['success_rate'] * (metrics['tasks_completed'] - 1) + 1) 
                         / metrics['tasks_completed'])
        else:
            new_success = ((metrics['success_rate'] * (metrics['tasks_completed'] - 1)) 
                         / metrics['tasks_completed'])
        metrics['success_rate'] = new_success
        
        # Update completion time
        new_avg_time = ((metrics['avg_completion_time'] * (metrics['tasks_completed'] - 1) +
                       task_result['completion_time']) / metrics['tasks_completed'])
        metrics['avg_completion_time'] = new_avg_time
        
        # Update collaboration score
        collab_edges = self.collaboration_graph.edges(agent_name, data=True)
        total_collabs = sum(edge[2]['weight'] for edge in collab_edges)
        metrics['collaboration_score'] = total_collabs / metrics['tasks_completed']

    def get_agent_analytics(self, agent_name):
        """Returns detailed analytics for an agent."""
        return {
            'performance_metrics': self.performance_metrics[agent_name],
            'collaboration_network': dict(self.collaboration_graph[agent_name]),
            'roles': [role for role, agents in self.agent_roles.items() 
                     if agent_name in agents],
            'current_state': self.agent_states[agent_name].tolist()
        }

    def optimize_team_composition(self):
        """Optimizes team compositions based on historical performance."""
        # Calculate agent compatibility scores
        compatibility_scores = {}
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 != agent2:
                    edge_data = self.collaboration_graph.get_edge_data(
                        agent1.name, agent2.name)
                    if edge_data:
                        score = (edge_data['weight'] * 
                               self.performance_metrics[agent1.name]['success_rate'] *
                               self.performance_metrics[agent2.name]['success_rate'])
                        compatibility_scores[(agent1.name, agent2.name)] = score
        
        return compatibility_scores

    def start_task_queue_monitor(self):
        """Starts monitoring the task queue in a separate thread."""
        threading.Thread(target=self.process_task_queue, daemon=True).start()

    def process_task_queue(self):
        """Processes queued tasks when agents become available."""
        while True:
            if not self.task_queue.empty():
                for agent in self.agents:
                    if agent.status == "Idle":
                        task = self.task_queue.get()
                        if task.is_ready(self.completed_tasks):
                            threading.Thread(target=agent.assign_task, args=(task, self)).start()
                        else:
                            print(f"⏳ Task '{task.name}' is still waiting for dependencies.")

# --- Example Usage ---
if __name__ == "__main__":
    manager = AgentManager()

    # Register agents with specific skills
    manager.register_agent("Agent Alpha", capabilities=["data_processing"], role="Data Scientist")
    manager.register_agent("Agent Beta", capabilities=["ml_training"], role="ML Engineer")
    manager.register_agent("Agent Gamma", capabilities=["reporting", "data_analysis"], role="Data Analyst")

    # Create tasks with dependencies
    task1 = Task(name="Data Cleaning", priority=2, duration=2, required_capabilities=["data_processing"], required_roles=["Data Scientist"], task_state=torch.randn(128))
    task2 = Task(name="Model Training", priority=1, duration=3, dependencies=["Data Cleaning"], required_capabilities=["ml_training"], required_roles=["ML Engineer"], task_state=torch.randn(128))
    task3 = Task(name="Report Summarization", priority=3, dependencies=["Model Training"], required_capabilities=["reporting"], required_roles=["Data Analyst"], task_state=torch.randn(128))
    task4 = Task(name="Anomaly Detection", priority=2, dependencies=["Data Cleaning"], required_capabilities=["data_analysis"], required_roles=["Data Analyst"], task_state=torch.randn(128))

    # Assign tasks
    manager.assign_task(task1)
    manager.assign_task(task2)
    manager.assign_task(task3)
    manager.assign_task(task4)

    # Start monitoring the task queue
    manager.start_task_queue_monitor()

import threading
import time
from queue import Queue
from collections import defaultdict
import torch

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
    
    def __init__(self, name, priority=1, duration=2, dependencies=None, required_capabilities=None, task_state=None):
        """
        Initializes a task.

        Args:
            name (str): Task name.
            priority (int): Task priority (higher value = higher priority).
            duration (int): Time (seconds) the task takes to complete.
            dependencies (list): List of dependent task names.
            required_capabilities (list): List of capabilities needed.
            task_state (torch.Tensor): Neural state of the task
        """
        self.name = name
        self.priority = priority
        self.duration = duration
        self.dependencies = dependencies or []
        self.required_capabilities = required_capabilities or []
        self.completed = False
        self.task_state = task_state

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

    def register_agent(self, name, capabilities=None):
        """Registers a new agent."""
        agent = Agent(name, capabilities, self.embedding_dim)
        self.agents.append(agent)
        print(f"🆕 Registered Agent: {name}")

    def get_available_agent(self, task_state=None):
        """
        Gets an available agent considering task state compatibility.
        
        Args:
            task_state (torch.Tensor, optional): Neural state of the task
            
        Returns:
            Agent: Available agent with highest compatibility
        """
        available_agents = [a for a in self.agents if a.status == "Idle"]
        
        if not available_agents:
            return None
            
        if task_state is None or not self.dynamic:
            return available_agents[0]
            
        # Find agent with most compatible state
        compatibility_scores = [
            torch.cosine_similarity(a.state_embedding.unsqueeze(0), task_state.unsqueeze(0))
            for a in available_agents
        ]
        
        best_agent_idx = torch.argmax(torch.tensor(compatibility_scores))
        best_agent = available_agents[best_agent_idx]
        
        # Update agent state
        best_agent.update_state(task_state)
        
        return best_agent

    def assign_task(self, task):
        """Assigns a task to an available agent with the required capabilities."""
        with self.lock:
            if not task.is_ready(self.completed_tasks):
                print(f"⏳ Task '{task.name}' is waiting for dependencies: {task.dependencies}")
                return

            agent = self.get_available_agent(task.task_state)
            if agent is not None:
                threading.Thread(target=agent.assign_task, args=(task, self)).start()
                return
            
            print(f"🔄 No suitable idle agents available, queuing task: {task.name}")
            self.task_queue.put(task)

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

    def mark_task_completed(self, task):
        """Marks a task as completed."""
        with self.lock:
            self.completed_tasks.add(task.name)
            print(f"🎯 Task '{task.name}' marked as completed.")

    def start_task_queue_monitor(self):
        """Starts monitoring the task queue in a separate thread."""
        threading.Thread(target=self.process_task_queue, daemon=True).start()

# --- Example Usage ---
if __name__ == "__main__":
    manager = AgentManager()

    # Register agents with specific skills
    manager.register_agent("Agent Alpha", capabilities=["data_processing"])
    manager.register_agent("Agent Beta", capabilities=["ml_training"])
    manager.register_agent("Agent Gamma", capabilities=["reporting", "data_analysis"])

    # Create tasks with dependencies
    task1 = Task(name="Data Cleaning", priority=2, duration=2, required_capabilities=["data_processing"], task_state=torch.randn(128))
    task2 = Task(name="Model Training", priority=1, duration=3, dependencies=["Data Cleaning"], required_capabilities=["ml_training"], task_state=torch.randn(128))
    task3 = Task(name="Report Summarization", priority=3, dependencies=["Model Training"], required_capabilities=["reporting"], task_state=torch.randn(128))
    task4 = Task(name="Anomaly Detection", priority=2, dependencies=["Data Cleaning"], required_capabilities=["data_analysis"], task_state=torch.randn(128))

    # Assign tasks
    manager.assign_task(task1)
    manager.assign_task(task2)
    manager.assign_task(task3)
    manager.assign_task(task4)

    # Start monitoring the task queue
    manager.start_task_queue_monitor()

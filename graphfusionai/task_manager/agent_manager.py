import threading
import time
from queue import Queue
from collections import defaultdict

class Agent:
    """Represents an AI Agent that executes tasks."""
    
    def __init__(self, name, capabilities=None):
        """
        Initializes an Agent.

        Args:
            name (str): The agent's name.
            capabilities (list): List of skills the agent can handle.
        """
        self.name = name
        self.capabilities = capabilities or []
        self.status = "Idle"
        self.current_task = None

    def assign_task(self, task):
        """Assigns a task to the agent."""
        if self.status == "Idle":
            self.current_task = task
            self.status = "Busy"
            print(f"🚀 {self.name} started task: {task.name} (Priority: {task.priority})")
            time.sleep(task.duration)  # Simulate task execution
            self.complete_task()
        else:
            print(f"⚠️ {self.name} is busy with another task.")

    def complete_task(self):
        """Marks task as completed."""
        print(f"✅ {self.name} completed task: {self.current_task.name}")
        self.current_task.mark_completed()
        self.current_task = None
        self.status = "Idle"

class Task:
    """Represents a task with dependencies and required capabilities."""
    
    def __init__(self, name, priority=1, duration=2, dependencies=None, required_capabilities=None):
        """
        Initializes a task.

        Args:
            name (str): Task name.
            priority (int): Task priority (higher value = higher priority).
            duration (int): Time (seconds) the task takes to complete.
            dependencies (list): List of dependent task names.
            required_capabilities (list): List of capabilities needed.
        """
        self.name = name
        self.priority = priority
        self.duration = duration
        self.dependencies = dependencies or []
        self.required_capabilities = required_capabilities or []
        self.completed = False

    def is_ready(self, completed_tasks):
        """Checks if all dependencies are met before execution."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_completed(self):
        """Marks the task as completed."""
        self.completed = True

class AgentManager:
    """Manages multiple AI agents and task assignments."""
    
    def __init__(self):
        self.agents = []
        self.task_queue = Queue()
        self.task_dependencies = defaultdict(list)
        self.completed_tasks = set()
        self.lock = threading.Lock()

    def register_agent(self, name, capabilities=None):
        """Registers a new agent."""
        agent = Agent(name, capabilities)
        self.agents.append(agent)
        print(f"🆕 Registered Agent: {name}")

    def list_agents(self):
        """Lists all registered agents."""
        return [(agent.name, agent.status, agent.capabilities) for agent in self.agents]

    def assign_task(self, task):
        """Assigns a task to an available agent with the required capabilities."""
        with self.lock:
            if not task.is_ready(self.completed_tasks):
                print(f"⏳ Task '{task.name}' is waiting for dependencies: {task.dependencies}")
                self.task_dependencies[task.name] = task
                return

            for agent in self.agents:
                if agent.status == "Idle" and any(skill in agent.capabilities for skill in task.required_capabilities):
                    threading.Thread(target=agent.assign_task, args=(task,)).start()
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
                            threading.Thread(target=agent.assign_task, args=(task,)).start()
                        else:
                            print(f"⏳ Task '{task.name}' is still waiting for dependencies.")

    def mark_task_completed(self, task):
        """Marks a task as completed and checks if dependent tasks can start."""
        with self.lock:
            self.completed_tasks.add(task.name)
            print(f"🎯 Task '{task.name}' marked as completed.")

        for dependent_task_name in list(self.task_dependencies.keys()):
            dependent_task = self.task_dependencies[dependent_task_name]
            if dependent_task.is_ready(self.completed_tasks):
                print(f"🚀 Now executing dependent task: {dependent_task.name}")
                self.assign_task(self.task_dependencies.pop(dependent_task_name))


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
    task1 = Task(name="Data Cleaning", priority=2, duration=2, required_capabilities=["data_processing"])
    task2 = Task(name="Model Training", priority=1, duration=3, dependencies=["Data Cleaning"], required_capabilities=["ml_training"])
    task3 = Task(name="Report Summarization", priority=3, dependencies=["Model Training"], required_capabilities=["reporting"])
    task4 = Task(name="Anomaly Detection", priority=2, dependencies=["Data Cleaning"], required_capabilities=["data_analysis"])

    # Assign tasks
    manager.assign_task(task1)
    manager.assign_task(task2)
    manager.assign_task(task3)
    manager.assign_task(task4)

    # Start monitoring the task queue
    manager.start_task_queue_monitor()

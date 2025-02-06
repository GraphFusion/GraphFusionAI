import threading
import time
from queue import Queue

class Agent:
    """Represents an AI Agent that executes tasks."""
    
    def __init__(self, name, capabilities=None):
        """
        Initializes an Agent.

        Args:
            name (str): The agent's name.
            capabilities (list): List of tasks the agent can handle.
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
            print(f"🚀 {self.name} started task: {task}")
            time.sleep(2)  
            self.complete_task()
        else:
            print(f"⚠️ {self.name} is busy with another task.")

    def complete_task(self):
        """Marks task as completed."""
        print(f"✅ {self.name} completed task: {self.current_task}")
        self.current_task = None
        self.status = "Idle"

class AgentManager:
    """Manages multiple AI agents and task assignments."""
    
    def __init__(self):
        self.agents = []
        self.task_queue = Queue()
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
        """Assigns a task to an available agent."""
        with self.lock:
            for agent in self.agents:
                if agent.status == "Idle":
                    threading.Thread(target=agent.assign_task, args=(task,)).start()
                    return
            print(f"🔄 No idle agents available, queuing task: {task}")
            self.task_queue.put(task)

    def process_task_queue(self):
        """Processes queued tasks when agents become available."""
        while True:
            if not self.task_queue.empty():
                for agent in self.agents:
                    if agent.status == "Idle":
                        task = self.task_queue.get()
                        threading.Thread(target=agent.assign_task, args=(task,)).start()

    def start_task_queue_monitor(self):
        """Starts monitoring the task queue in a separate thread."""
        threading.Thread(target=self.process_task_queue, daemon=True).start()

import threading
import time
from queue import PriorityQueue

class Agent:
    """Represents an AI agent capable of executing tasks."""
    
    def __init__(self, name, capabilities=None):
        """
        Initializes an agent.

        Args:
            name (str): The agent's name.
            capabilities (list, optional): Tasks the agent can handle.
        """
        self.name = name
        self.capabilities = capabilities or []
        self.status = "Idle"
        self.current_task = None

    def assign_task(self, task):
        """Assigns a task to the agent if available."""
        if self.status == "Idle":
            self.current_task = task
            self.status = "Busy"
            print(f"🚀 {self.name} started task: {task['name']} (Priority: {task['priority']})")
            time.sleep(task['duration'])  
            self.complete_task()
        else:
            print(f"⚠️ {self.name} is busy with another task.")

    def complete_task(self):
        """Marks the task as completed."""
        print(f"✅ {self.name} completed task: {self.current_task['name']}")
        self.current_task = None
        self.status = "Idle"

class AgentManager:
    """Manages agents and dynamically assigns tasks based on priority and availability."""
    
    def __init__(self):
        self.agents = []
        self.task_queue = PriorityQueue()  
        self.lock = threading.Lock()

    def register_agent(self, name, capabilities=None):
        """Registers a new agent."""
        agent = Agent(name, capabilities)
        self.agents.append(agent)
        print(f"🆕 Registered Agent: {name}")

    def list_agents(self):
        """Lists all registered agents."""
        return [(agent.name, agent.status, agent.capabilities) for agent in self.agents]

    def assign_task(self, task_name, duration=2, priority=1):
        """
        Assigns a task to an available agent or queues it.

        Args:
            task_name (str): Task description.
            duration (int, optional): Simulated task execution time (default: 2 sec).
            priority (int, optional): Task priority (lower values = higher priority).
        """
        task = {"name": task_name, "duration": duration, "priority": priority}
        
        with self.lock:
            for agent in self.agents:
                if agent.status == "Idle":
                    threading.Thread(target=agent.assign_task, args=(task,)).start()
                    return
            
            print(f"🔄 No idle agents available, queuing task: {task_name}")
            self.task_queue.put((priority, task))

    def process_task_queue(self):
        """Continuously assigns queued tasks when agents become available."""
        while True:
            if not self.task_queue.empty():
                with self.lock:
                    priority, task = self.task_queue.get()
                    for agent in self.agents:
                        if agent.status == "Idle":
                            print(f"📌 Assigning queued task: {task['name']} to {agent.name}")
                            threading.Thread(target=agent.assign_task, args=(task,)).start()
                            break
            time.sleep(1)  

    def start_task_queue_monitor(self):
        """Starts monitoring the task queue in a background thread."""
        threading.Thread(target=self.process_task_queue, daemon=True).start()


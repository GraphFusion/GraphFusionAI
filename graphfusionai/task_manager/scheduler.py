import time
import threading
from task_graph import TaskGraph
from agent_manager import AgentManager

class Scheduler:
    """
    The Scheduler is responsible for managing task execution. It schedules tasks based on their
    dependencies and ensures that tasks are executed in the correct order. It assigns tasks to 
    agents and can manage concurrency and timing.
    """
    
    def __init__(self, agent_manager, max_concurrent_tasks=2):
        """
        Initializes the Scheduler.
        
        Args:
            agent_manager (AgentManager): The manager for agents who will execute tasks.
            max_concurrent_tasks (int): The maximum number of tasks to run concurrently.
        """
        self.agent_manager = agent_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_status = {}  
        self.lock = threading.Lock()  
    
    def assign_task_to_agent(self, task):
        """Assigns a task to an available agent."""
        agent = self.agent_manager.get_available_agent()
        if agent:
            print(f"Assigning task '{task}' to agent {agent.name}.")
            agent.execute_task(task)
            self.task_status[task] = 'Running'
        else:
            print(f"No available agents for task '{task}', retrying...")
            time.sleep(2)  # Wait before retrying

    def execute_task(self, task):
        """Executes a single task."""
        with self.lock:
            if self.task_status.get(task) != 'Running':
                self.task_status[task] = 'Pending'
                print(f"Task '{task}' is pending.")
                self.assign_task_to_agent(task)

    def execute_concurrent_tasks(self, tasks):
        """Executes tasks concurrently, respecting the max concurrency limit."""
        running_tasks = []
        
        # Ensure that we are not exceeding max concurrency limit
        for task in tasks:
            if len(running_tasks) >= self.max_concurrent_tasks:
                # Wait for one of the running tasks to complete
                completed_task = running_tasks.pop(0)
                self.task_status[completed_task] = 'Completed'
            
            # Execute the task
            running_tasks.append(task)
            self.execute_task(task)
        
        # Wait for remaining tasks to complete
        while running_tasks:
            task = running_tasks.pop(0)
            self.task_status[task] = 'Completed'
            print(f"Task '{task}' is completed.")
    
    def execute_task_sequence(self, tasks):
        """Executes tasks in sequence, respecting their dependencies."""
        for task in tasks:
            self.execute_task(task)
            print(f"Task '{task}' is completed.")

    def run_schedule(self, task_graph):
        """Runs the task schedule based on the task graph and dependencies."""
        ordered_tasks = task_graph.topological_sort()  # Get tasks in execution order
        self.execute_concurrent_tasks(ordered_tasks)

    def get_task_status(self, task):
        """Gets the current status of a task."""
        return self.task_status.get(task, 'Not Started')

    def log_status(self):
        """Logs the status of all tasks."""
        for task, status in self.task_status.items():
            print(f"Task '{task}' status: {status}")

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .task_graph import TaskGraph
from .task_executor import TaskExecutor
from .scheduler import Scheduler
from .agent_manager import AgentManager

class TaskManager:
    """Graph-powered Task Manager for AI Agents."""

    def __init__(self):
        self.task_graph = TaskGraph()
        self.task_executor = TaskExecutor()
        self.scheduler = Scheduler(self.task_graph)
        self.agent_manager = AgentManager()

    def create_task(self, name, priority="medium", dependencies=None):
        """Creates a task and adds it to the task graph."""
        return self.task_graph.add_task(name, priority, dependencies)

    def assign_task(self, task, agent_name):
        """Assigns a task to an available agent."""
        agent = self.agent_manager.get_agent(agent_name)
        if agent:
            agent.assign(task)
            return agent
        return None

    def execute_task(self, task):
        """Executes a task through an assigned agent."""
        return self.task_executor.run(task)

    def graph_powered_task(self):
        """Returns a unified function for graph-powered task execution."""
        def run(task_name, agent_name):
            task = self.create_task(task_name)
            agent = self.assign_task(task, agent_name)
            if agent:
                return self.execute_task(task)
            return f"No available agent for {task_name}"
        
        return run

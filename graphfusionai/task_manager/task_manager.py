import os
import sys
import logging
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from task_graph import TaskGraph
from task_executor import TaskExecutor
from scheduler import Scheduler
from agent_manager import AgentManager

class TaskManager:
    """Graph-powered task manager for orchestrating AI agent workflows."""
    
    def __init__(self, dynamic_agents=True, enable_scheduling=True):
        """
        Initializes the TaskManager with integrated components.

        Args:
            dynamic_agents (bool): If True, agents are dynamically assigned.
            enable_scheduling (bool): If True, enables task scheduling.
        """
        self.task_graph = TaskGraph()
        self.agent_manager = AgentManager(dynamic=dynamic_agents)
        self.executor = TaskExecutor()
        self.scheduler = Scheduler(agent_manager=self.agent_manager, task_graph=self.task_graph) if enable_scheduling else None

    def add_task(self, task_name, dependencies=None, priority=0):
        """
        Adds a task to the system with optional dependencies and priority.

        Args:
            task_name (str): The name of the task.
            dependencies (list, optional): List of dependent tasks.
            priority (int, optional): Task priority (lower value = higher priority).
        """
        self.task_graph.add_task(task_name)
        if dependencies:
            for dep in dependencies:
                self.task_graph.add_dependency(task_name, dep)
        if self.scheduler:
            self.scheduler.add_task(task_name, priority)

    def run_task(self, task_name):
        """
        Runs a single task by assigning it to an agent.

        Args:
            task_name (str): The task to execute.
        """
        agent = self.agent_manager.get_available_agent()
        if agent:
            self.executor.execute(agent, task_name)
        else:
            print(f"No available agents for '{task_name}', retrying...")
            self.scheduler.schedule_task(task_name) if self.scheduler else None

    def execute_all(self):
        """Executes all tasks based on dependencies and scheduling."""
        if self.scheduler:
            self.scheduler.execute_all()
        else:
            execution_order = self.task_graph.topological_sort()
            for task in execution_order:
                self.run_task(task)

    def visualize(self):
        """Displays the task graph and scheduled tasks."""
        self.task_graph.visualize()
        if self.scheduler:
            self.scheduler.visualize_schedule()

    @staticmethod
    def graph_powered_task():
        """
        Returns an instance of TaskManager for easy import and use.
        
        Example:
            from graphfusionai import TaskManager
            task_manager = TaskManager.graph_powered_task()
        """
        return TaskManager()
    

# Example Usage
# Initialize Task Manager
task_manager = TaskManager.graph_powered_task()

# Add tasks with dependencies
task_manager.add_task("Data Preprocessing")
task_manager.add_task("Feature Engineering", dependencies=["Data Preprocessing"])
task_manager.add_task("Model Training", dependencies=["Feature Engineering"])
task_manager.add_task("Model Evaluation", dependencies=["Model Training"], priority=1)

# Run all tasks
task_manager.execute_all()

# Visualize dependencies
task_manager.visualize()



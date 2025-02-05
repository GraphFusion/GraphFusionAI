import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from task_manager import TaskManager

class TaskExecutor:
    def __init__(self, task_manager, agent_manager):
        """Initialize the TaskExecutor with TaskManager and AgentManager."""
        self.task_manager = task_manager
        self.agent_manager = agent_manager

    def execute(self, task_id, required_skills):
        """Executes a task by assigning it to an appropriate agent."""
        assigned_agent = self.agent_manager.assign_task(task_id, required_skills)
        if assigned_agent:
            print(f"Task {task_id} is being processed by Agent {assigned_agent}.")
        else:
            print(f"Failed to assign task {task_id}.")

    def run(self):
        """Runs all executable tasks in order."""
        while True:
            tasks_to_run = self.task_manager.get_next_tasks()
            if not tasks_to_run:
                break
            for task in tasks_to_run:
                self.execute_task(task)


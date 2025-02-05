import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from task_manager import TaskManager

class TaskExecutor:
    def __init__(self, task_manager):
        """Initialize with a Task Manager instance."""
        self.task_manager = task_manager

    def execute_task(self, task_id):
        """Simulate task execution and mark as complete."""
        print(f"Executing task: {task_id}")
        time.sleep(2)  
        self.task_manager.update_task_status(task_id, "Completed")
        self.task_manager.task_graph.mark_task_completed(task_id)
        print(f"Task {task_id} completed.")

    def run(self):
        """Runs all executable tasks in order."""
        while True:
            tasks_to_run = self.task_manager.get_next_tasks()
            if not tasks_to_run:
                break
            for task in tasks_to_run:
                self.execute_task(task)


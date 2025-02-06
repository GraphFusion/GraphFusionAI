import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
from queue import Queue
from task_graph import TaskGraph

class TaskExecutor:
    """Executes tasks in order while managing dependencies and failures."""

    def __init__(self, task_graph, max_retries=3, parallel=True):
        """
        Initializes the TaskExecutor.

        Args:
            task_graph (TaskGraph): The task dependency graph.
            max_retries (int): Number of times to retry a failed task.
            parallel (bool): Whether to execute independent tasks in parallel.
        """
        self.task_graph = task_graph
        self.max_retries = max_retries
        self.parallel = parallel
        self.task_status = {}  # Tracks task statuses

    def execute_task(self, task_name):
        """
        Executes an individual task.

        Args:
            task_name (str): Name of the task.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"🔄 Executing Task: {task_name} (Attempt {retries+1})")
                time.sleep(1)  # Simulate task execution time
                self.task_status[task_name] = "Completed"
                print(f"✅ Task Completed: {task_name}")
                return True
            except Exception as e:
                print(f"❌ Task Failed: {task_name}, Error: {e}")
                retries += 1

        self.task_status[task_name] = "Failed"
        print(f"⚠️ Task {task_name} failed after {self.max_retries} retries.")
        return False

    def run_sequential(self, tasks):
        """
        Runs tasks sequentially in execution order.

        Args:
            tasks (list): Ordered list of tasks.
        """
        for task in tasks:
            self.execute_task(task)

    def run_parallel(self, tasks):
        """
        Runs tasks in parallel if they are independent.

        Args:
            tasks (list): Ordered list of tasks.
        """
        task_queue = Queue()
        threads = []

        for task in tasks:
            task_queue.put(task)

        def worker():
            while not task_queue.empty():
                task = task_queue.get()
                self.execute_task(task)
                task_queue.task_done()

        for _ in range(min(len(tasks), 4)):  # Limit parallel threads
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def run(self):
        """
        Executes tasks in the correct order while respecting dependencies.
        """
        tasks = self.task_graph.get_execution_order()
        print(f"📌 Execution Order: {tasks}")

        if self.parallel:
            self.run_parallel(tasks)
        else:
            self.run_sequential(tasks)

        print("🎯 All tasks executed!")


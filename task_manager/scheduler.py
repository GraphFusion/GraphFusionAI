import time
import threading
from task_manager import TaskManager
from task_executor import TaskExecutor
from agent_manager import AgentManager

class TaskScheduler:
    def __init__(self, task_manager, task_executor):
        """Initialize scheduler with Task Manager and Task Executor."""
        self.task_manager = task_manager
        self.task_executor = task_executor
        self.running = False

    def get_runnable_tasks(self):
        """Fetches tasks that have no pending dependencies and are ready to run."""
        return self.task_manager.get_next_tasks()

    def schedule_tasks(self):
        """Continuously schedules and executes tasks as they become ready."""
        self.running = True
        while self.running:
            runnable_tasks = self.get_runnable_tasks()

            if not runnable_tasks:
                time.sleep(2)  # No ready tasks, wait before checking again.
                continue

            for task_id in runnable_tasks:
                self.task_executor.execute_task(task_id)

            time.sleep(1)  # Short pause to prevent excessive looping.

    def start(self):
        """Runs the scheduler in a separate thread."""
        scheduler_thread = threading.Thread(target=self.schedule_tasks)
        scheduler_thread.daemon = True  # Allows graceful shutdown
        scheduler_thread.start()
        print("Task Scheduler started...")

    def stop(self):
        """Stops the scheduler loop."""
        self.running = False
        print("Task Scheduler stopped.")

# Example usage
if __name__ == "__main__":
    tm = TaskManager()
    am = AgentManager()
    am.register_agent("agent1", ["ML", "Data Processing"])

    tm.add_task("task1")
    tm.add_task("task2", dependencies=["task1"])
    tm.add_task("task3", dependencies=["task2"])

    executor = TaskExecutor(tm, am)
    scheduler = TaskScheduler(tm, executor)

    scheduler.start()

    # Simulate running the scheduler for 10 seconds
    time.sleep(10)
    scheduler.stop()

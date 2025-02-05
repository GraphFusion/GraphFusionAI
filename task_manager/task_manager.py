from task_graph import TaskGraph

class TaskManager:
    def __init__(self):
        """Initialize the Task Manager with a task graph."""
        self.task_graph = TaskGraph()
        self.tasks = {}  

    def add_task(self, task_id, dependencies=None):
        """Adds a new task to the task graph with optional dependencies."""
        self.task_graph.add_task(task_id, dependencies)
        self.tasks[task_id] = "Pending"

    def update_task_status(self, task_id, status):
        """Updates the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id] = status
        else:
            raise ValueError(f"Task {task_id} not found")

    def get_task_status(self, task_id):
        """Retrieves the status of a given task."""
        return self.tasks.get(task_id, "Not Found")

    def get_next_tasks(self):
        """Returns tasks that are ready for execution (no pending dependencies)."""
        return self.task_graph.get_executable_tasks()



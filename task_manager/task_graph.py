import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networkx as nx

class TaskGraph:
    def __init__(self):
        """Initialize a directed graph for task dependencies."""
        self.graph = nx.DiGraph()

    def add_task(self, task_id, dependencies=None):
        """Adds a task to the graph and sets up dependencies."""
        self.graph.add_node(task_id)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, task_id)  

    def get_executable_tasks(self):
        """Returns tasks that have no dependencies or all dependencies are completed."""
        executable = []
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:  
                executable.append(node)
        return executable

    def mark_task_completed(self, task_id):
        """Removes completed task from the graph."""
        if task_id in self.graph:
            self.graph.remove_node(task_id)

# Example usage
if __name__ == "__main__":
    tg = TaskGraph()
    tg.add_task("task1")
    tg.add_task("task2", dependencies=["task1"])
    
    print(tg.get_executable_tasks())  
    tg.mark_task_completed("task1")
    print(tg.get_executable_tasks())  
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx

class TaskGraph:
    """Graph structure to manage task dependencies."""

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for task dependencies

    def add_task(self, task_name, priority="medium", dependencies=None):
        """
        Adds a task to the graph with optional dependencies.

        Args:
            task_name (str): Name of the task.
            priority (str): Priority level (low, medium, high).
            dependencies (list, optional): List of dependent task names.

        Returns:
            str: The task name.
        """
        self.graph.add_node(task_name, priority=priority)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, task_name)  # Task depends on another
        return task_name

    def get_execution_order(self):
        """
        Returns tasks in topological order to ensure dependencies are met.

        Returns:
            list: Ordered list of tasks for execution.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Cycle detected in task dependencies!")
        return list(nx.topological_sort(self.graph))

    def get_task_priority(self, task_name):
        """
        Retrieves the priority level of a task.

        Args:
            task_name (str): Name of the task.

        Returns:
            str: Priority level (low, medium, high).
        """
        return self.graph.nodes[task_name].get("priority", "medium")

    def visualize_graph(self):
        """
        Visualizes the task graph (for debugging and understanding dependencies).
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
        plt.title("Task Dependency Graph")
        plt.show()

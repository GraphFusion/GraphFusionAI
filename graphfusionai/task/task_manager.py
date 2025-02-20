"""
Task management and execution coordination.
"""
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import networkx as nx
from .task import Task, TaskStatus, TaskType
from .task_queue import TaskQueue
from .task_scheduler import TaskScheduler
from .task_executor import TaskExecutor
from ..agents import BaseAgent
from ..memory import MemoryManager
from ..knowledge_graph import KnowledgeGraph

class TaskManager:
    """
    Manages task lifecycle and execution.
    
    Features:
    - Task dependency management
    - Dynamic task scheduling
    - Parallel execution
    - Resource allocation
    - Performance monitoring
    - Failure recovery
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None
    ):
        """Initialize task manager."""
        self.tasks: Dict[str, Task] = {}
        self.task_graph = nx.DiGraph()  # Dependency graph
        self.execution_history: List[Dict] = []
        
        # Components
        self.queue = TaskQueue()
        self.scheduler = TaskScheduler()
        self.executor = TaskExecutor()
        self.memory = memory or MemoryManager()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0.0,
            "success_rate": 0.0,
            "parallel_tasks": 0
        }
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to be managed.
        
        Args:
            task: Task to add
        """
        self.tasks[task.id] = task
        
        # Update dependency graph
        self.task_graph.add_node(task.id)
        for dep_id in task.dependencies:
            self.task_graph.add_edge(dep_id, task.id)
        
        # Add to queue if ready
        if task.is_ready():
            self.queue.add(task)
    
    def execute_tasks(
        self,
        agents: Dict[str, BaseAgent],
        max_parallel: int = 10
    ) -> Dict[str, Any]:
        """
        Execute all tasks with available agents.
        
        Args:
            agents: Available agents
            max_parallel: Maximum parallel tasks
            
        Returns:
            Execution results and metrics
        """
        results = {
            "completed": [],
            "failed": [],
            "skipped": [],
            "metrics": self.metrics
        }
        
        while True:
            # Get ready tasks up to max_parallel
            ready_tasks = self.get_ready_tasks(max_parallel)
            if not ready_tasks:
                break
            
            # Update parallel execution metric
            self.metrics["parallel_tasks"] = max(
                self.metrics["parallel_tasks"],
                len(ready_tasks)
            )
            
            # Execute ready tasks
            for task in ready_tasks:
                # Find best agent
                agent = self.scheduler.assign_agent(task, agents)
                if not agent:
                    task.skip()
                    results["skipped"].append(task.id)
                    continue
                
                # Execute task
                execution_result = self.executor.execute_task(
                    task,
                    agent,
                    self.memory,
                    self.knowledge_graph
                )
                
                # Handle result
                if execution_result["status"] == TaskStatus.COMPLETED:
                    self._handle_completion(task, execution_result, results)
                else:
                    self._handle_failure(task, execution_result, results)
                
                # Record execution
                self.execution_history.append({
                    "task_id": task.id,
                    "agent_id": agent.id,
                    "result": execution_result,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update final metrics
        self._update_metrics()
        results["metrics"] = self.metrics
        
        return results
    
    def get_ready_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """
        Get tasks ready for execution.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of ready tasks
        """
        ready_tasks = []
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                self._check_dependencies_met(task)):
                ready_tasks.append(task)
                if limit and len(ready_tasks) >= limit:
                    break
        return ready_tasks
    
    def _check_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are met."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _handle_completion(
        self,
        task: Task,
        result: Dict[str, Any],
        execution_results: Dict[str, Any]
    ) -> None:
        """Handle successful task completion."""
        task.complete(result)
        execution_results["completed"].append(task.id)
        self.metrics["tasks_completed"] += 1
        
        # Update knowledge graph
        self.knowledge_graph.add_node(
            task.id,
            type="task",
            status="completed",
            metadata=task.metadata
        )
        
        # Add completion memory
        self.memory.add_memory({
            "type": "task_completion",
            "task_id": task.id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update dependent tasks
        for successor in self.task_graph.successors(task.id):
            self.tasks[successor].remove_dependency(task.id)
            if self.tasks[successor].is_ready():
                self.queue.add(self.tasks[successor])
    
    def _handle_failure(
        self,
        task: Task,
        result: Dict[str, Any],
        execution_results: Dict[str, Any]
    ) -> None:
        """Handle task failure."""
        task.fail(result.get("error"))
        execution_results["failed"].append(task.id)
        self.metrics["tasks_failed"] += 1
        
        # Update knowledge graph
        self.knowledge_graph.add_node(
            task.id,
            type="task",
            status="failed",
            error=result.get("error"),
            metadata=task.metadata
        )
        
        # Add failure memory
        self.memory.add_memory({
            "type": "task_failure",
            "task_id": task.id,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle failure based on strategy
        if task.failure_strategy == "retry":
            task.reset()
            self.queue.add(task)
        
        elif task.failure_strategy == "skip_dependent":
            for successor in self.task_graph.successors(task.id):
                self.tasks[successor].skip()
                execution_results["skipped"].append(successor)
        
        elif task.failure_strategy == "alternate":
            if task.alternate_task:
                alt_task = Task.from_dict(task.alternate_task)
                self.add_task(alt_task)
    
    def _update_metrics(self) -> None:
        """Update overall performance metrics."""
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total_tasks > 0:
            self.metrics["success_rate"] = (
                self.metrics["tasks_completed"] / total_tasks
            )
            
        # Calculate average completion time
        completion_times = [
            task.metrics["completion_time"]
            for task in self.tasks.values()
            if task.metrics["completion_time"] is not None
        ]
        if completion_times:
            self.metrics["avg_completion_time"] = (
                sum(completion_times) / len(completion_times)
            )
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        return [
            task for task in self.tasks.values()
            if task.status == status
        ]
    
    def get_critical_path(self) -> List[str]:
        """Get critical path of tasks."""
        try:
            return nx.dag_longest_path(self.task_graph)
        except nx.NetworkXError:
            return []
    
    def get_task_dependencies(self, task_id: str) -> Set[str]:
        """Get all dependencies for a task."""
        try:
            return set(nx.ancestors(self.task_graph, task_id))
        except nx.NetworkXError:
            return set()
    
    def get_dependent_tasks(self, task_id: str) -> Set[str]:
        """Get all tasks that depend on this task."""
        try:
            return set(nx.descendants(self.task_graph, task_id))
        except nx.NetworkXError:
            return set()
    
    def visualize_task_graph(self, path: str) -> None:
        """
        Save visualization of task dependency graph.
        
        Args:
            path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Create position layout
        pos = nx.spring_layout(self.task_graph)
        
        # Draw nodes with colors based on status
        colors = {
            TaskStatus.PENDING: 'gray',
            TaskStatus.IN_PROGRESS: 'yellow',
            TaskStatus.COMPLETED: 'green',
            TaskStatus.FAILED: 'red',
            TaskStatus.BLOCKED: 'orange',
            TaskStatus.SKIPPED: 'blue'
        }
        
        for status in TaskStatus:
            nodes = [
                n for n in self.task_graph.nodes
                if self.tasks[n].status == status
            ]
            if nodes:
                nx.draw_networkx_nodes(
                    self.task_graph,
                    pos,
                    nodelist=nodes,
                    node_color=colors[status],
                    label=status.value
                )
        
        # Draw edges
        nx.draw_networkx_edges(self.task_graph, pos)
        
        # Draw labels
        labels = {
            n: f"{self.tasks[n].name}\n{self.tasks[n].status.value}"
            for n in self.task_graph.nodes
        }
        nx.draw_networkx_labels(self.task_graph, pos, labels)
        
        plt.title("Task Dependency Graph")
        plt.legend()
        plt.savefig(path)
        plt.close()

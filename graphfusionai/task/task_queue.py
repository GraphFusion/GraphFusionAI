"""
Priority-based task queue implementation.
"""
from typing import List, Dict, Any, Optional
from queue import PriorityQueue
from .task import Task, TaskStatus

class TaskQueue:
    """
    Priority queue for managing tasks.
    
    Features:
    - Priority-based ordering
    - Task filtering
    - Queue statistics
    - State persistence
    """
    
    def __init__(self):
        """Initialize task queue."""
        self._queue = PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._completed: List[str] = []
        
        # Queue statistics
        self.stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "avg_wait_time": 0.0
        }
    
    def add(self, task: Task) -> None:
        """
        Add task to queue.
        
        Args:
            task: Task to add
        """
        # Priority is negated because PriorityQueue is min-heap
        self._queue.put((-task.priority, task.id))
        self._tasks[task.id] = task
        self.stats["total_enqueued"] += 1
    
    def get(self) -> Optional[Task]:
        """
        Get next task from queue.
        
        Returns:
            Next task or None if queue is empty
        """
        while not self._queue.empty():
            _, task_id = self._queue.get()
            task = self._tasks[task_id]
            
            if task.status == TaskStatus.PENDING:
                self.stats["total_dequeued"] += 1
                return task
        
        return None
    
    def complete_task(self, task_id: str) -> None:
        """
        Mark task as completed.
        
        Args:
            task_id: ID of completed task
        """
        if task_id in self._tasks:
            self._completed.append(task_id)
            
            # Update dependent tasks
            task = self._tasks[task_id]
            for other_task in self._tasks.values():
                if task_id in other_task.dependencies:
                    other_task.remove_dependency(task_id)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    def get_completed(self) -> List[Task]:
        """Get all completed tasks."""
        return [
            self._tasks[task_id]
            for task_id in self._completed
        ]
    
    def clear(self) -> None:
        """Clear all tasks from queue."""
        self._queue = PriorityQueue()
        self._tasks.clear()
        self._completed.clear()
    
    def serialize(self) -> Dict[str, Any]:
        """Convert queue to serializable format."""
        return {
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self._tasks.items()
            },
            "completed": self._completed,
            "stats": self.stats
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load queue from serialized format."""
        self._tasks = {
            task_id: Task.from_dict(task_data)
            for task_id, task_data in data["tasks"].items()
        }
        self._completed = data["completed"]
        self.stats = data["stats"]
        
        # Rebuild priority queue
        self._queue = PriorityQueue()
        for task in self._tasks.values():
            if task.id not in self._completed:
                self._queue.put((-task.priority, task.id))

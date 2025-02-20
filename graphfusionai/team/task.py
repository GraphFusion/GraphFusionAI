"""
Task management for team-based operations.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from queue import PriorityQueue

class Task:
    """
    Represents a task that can be executed by team members.
    
    Features:
    - Step-by-step execution
    - Priority management
    - Progress tracking
    - Resource requirements
    - Dependencies
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        priority: int = 0,
        required_skills: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        resources: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None
    ):
        """
        Initialize a new task.
        
        Args:
            name: Task name
            description: Task description
            steps: List of execution steps
            priority: Task priority (higher = more important)
            required_skills: Required skills for the task
            dependencies: IDs of tasks that must complete first
            resources: Required resources
            deadline: Optional deadline
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.steps = steps
        self.priority = priority
        self.required_skills = required_skills or []
        self.dependencies = dependencies or []
        self.resources = resources or {}
        self.deadline = deadline
        
        # Status tracking
        self.status = "pending"  # pending, in_progress, completed, failed
        self.current_step = 0
        self.assigned_agents = []
        self.start_time = None
        self.end_time = None
        self.results = []
        
    def start(self) -> None:
        """Mark task as started."""
        self.status = "in_progress"
        self.start_time = datetime.now()
        
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark task as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
        self.results.append(result)
        
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.results.append({"error": error})
        
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
        
    def is_ready(self) -> bool:
        """Check if task is ready to execute."""
        return len(self.dependencies) == 0
        
    def remove_dependency(self, task_id: str) -> None:
        """Remove a dependency once it's completed."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "priority": self.priority,
            "required_skills": self.required_skills,
            "dependencies": self.dependencies,
            "resources": self.resources,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": self.results
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary format."""
        task = cls(
            name=data["name"],
            description=data["description"],
            steps=data["steps"],
            priority=data["priority"],
            required_skills=data["required_skills"],
            dependencies=data["dependencies"],
            resources=data["resources"],
            deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None
        )
        task.id = data["id"]
        task.status = data["status"]
        task.current_step = data["current_step"]
        if data["start_time"]:
            task.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            task.end_time = datetime.fromisoformat(data["end_time"])
        task.results = data["results"]
        return task

class TaskQueue:
    """
    Priority queue for managing team tasks.
    
    Features:
    - Priority-based ordering
    - Dependency tracking
    - Dynamic task insertion
    - Progress monitoring
    """
    
    def __init__(self):
        """Initialize task queue."""
        self._queue = PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._completed: List[str] = []
        
    def add(self, task: Task) -> None:
        """Add a task to the queue."""
        # Priority is negated because PriorityQueue is min-heap
        self._queue.put((-task.priority, task.id))
        self._tasks[task.id] = task
        
    def next(self) -> Optional[Task]:
        """Get next ready task with highest priority."""
        while not self._queue.empty():
            _, task_id = self._queue.get()
            task = self._tasks[task_id]
            
            if task.is_ready():
                return task
                
            # Put back if not ready
            self._queue.put((-task.priority, task_id))
            
        return None
        
    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self._tasks:
            self._completed.append(task_id)
            
            # Update dependencies
            for task in self._tasks.values():
                task.remove_dependency(task_id)
                
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
        
    def serialize(self) -> Dict[str, Any]:
        """Convert queue to serializable format."""
        return {
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self._tasks.items()
            },
            "completed": self._completed
        }
        
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load queue from serialized format."""
        self._tasks = {
            task_id: Task.from_dict(task_data)
            for task_id, task_data in data["tasks"].items()
        }
        self._completed = data["completed"]
        
        # Rebuild priority queue
        self._queue = PriorityQueue()
        for task in self._tasks.values():
            if task.id not in self._completed:
                self._queue.put((-task.priority, task.id))

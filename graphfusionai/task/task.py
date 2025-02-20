"""
Core task definitions for GraphFusionAI.
"""
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TaskType(Enum):
    """Types of tasks that can be executed."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DECISION = "decision"
    CUSTOM = "custom"

class Task:
    """
    Represents a task that can be executed by agents.
    
    Features:
    - Step-by-step execution
    - Conditional branching
    - Dynamic task generation
    - Failure handling strategies
    - Resource requirements
    - Performance tracking
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        steps: List[Dict[str, Any]],
        required_skills: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        deadline: Optional[datetime] = None,
        failure_strategy: str = "skip_dependent",
        alternate_task: Optional[Dict] = None,
        resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize task."""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.task_type = task_type
        self.steps = steps
        self.required_skills = required_skills or []
        self.dependencies = dependencies or []
        self.priority = priority
        self.deadline = deadline
        self.failure_strategy = failure_strategy
        self.alternate_task = alternate_task
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Execution state
        self.status = TaskStatus.PENDING
        self.current_step = 0
        self.assigned_agents = []
        self.start_time = None
        self.end_time = None
        self.results = []
        self.error = None
        
        # Performance metrics
        self.metrics = {
            "attempts": 0,
            "completion_time": None,
            "success_rate": 0.0
        }
    
    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.start_time = datetime.now()
        self.metrics["attempts"] += 1
    
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now()
        self.results.append(result)
        self._update_metrics(True)
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.end_time = datetime.now()
        self.error = error
        self._update_metrics(False)
    
    def skip(self) -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.end_time = datetime.now()
    
    def block(self) -> None:
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
    
    def reset(self) -> None:
        """Reset task for retry."""
        self.status = TaskStatus.PENDING
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        self.error = None
    
    def is_ready(self) -> bool:
        """Check if task is ready to execute."""
        return len(self.dependencies) == 0
    
    def remove_dependency(self, task_id: str) -> None:
        """Remove a dependency once it's completed."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def _update_metrics(self, success: bool) -> None:
        """Update task performance metrics."""
        # Update completion time
        duration = self.get_duration()
        if duration:
            self.metrics["completion_time"] = duration
        
        # Update success rate
        total_attempts = self.metrics["attempts"]
        current_success_rate = self.metrics["success_rate"]
        self.metrics["success_rate"] = (
            (current_success_rate * (total_attempts - 1) + int(success))
            / total_attempts
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "steps": self.steps,
            "required_skills": self.required_skills,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "failure_strategy": self.failure_strategy,
            "alternate_task": self.alternate_task,
            "resources": self.resources,
            "metadata": self.metadata,
            "status": self.status.value,
            "current_step": self.current_step,
            "assigned_agents": [agent.id for agent in self.assigned_agents],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": self.results,
            "error": self.error,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary format."""
        task = cls(
            name=data["name"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            steps=data["steps"],
            required_skills=data["required_skills"],
            dependencies=data["dependencies"],
            priority=data["priority"],
            deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None,
            failure_strategy=data["failure_strategy"],
            alternate_task=data["alternate_task"],
            resources=data["resources"],
            metadata=data["metadata"]
        )
        
        # Restore state
        task.id = data["id"]
        task.status = TaskStatus(data["status"])
        task.current_step = data["current_step"]
        if data["start_time"]:
            task.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            task.end_time = datetime.fromisoformat(data["end_time"])
        task.results = data["results"]
        task.error = data["error"]
        task.metrics = data["metrics"]
        
        return task
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks by priority."""
        return self.priority > other.priority  # Higher priority first

"""
Role definitions and management for team-based agents.
"""
from typing import List, Dict, Any, Optional
from graphfusionai.agents import BaseAgent

class Role:
    """
    Defines a role that agents can take within a team.
    
    Features:
    - Skill requirements
    - Responsibility definitions
    - Performance metrics
    - Hierarchical relationships
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        required_skills: Optional[List[str]] = None,
        responsibilities: Optional[List[str]] = None,
        parent_role: Optional['Role'] = None
    ):
        """
        Initialize a new role.
        
        Args:
            name: Role name
            description: Role description
            required_skills: List of required skills
            responsibilities: List of responsibilities
            parent_role: Optional parent role for hierarchy
        """
        self.name = name
        self.description = description
        self.required_skills = required_skills or []
        self.responsibilities = responsibilities or []
        self.parent_role = parent_role
        self.agents: List[BaseAgent] = []
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "avg_completion_time": 0.0
        }
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to this role."""
        if agent not in self.agents:
            self.agents.append(agent)
            agent.role = self
    
    def remove_agent(self, agent: BaseAgent) -> None:
        """Remove an agent from this role."""
        if agent in self.agents:
            self.agents.remove(agent)
            if agent.role == self:
                agent.role = None
    
    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this role can handle a specific task."""
        required_skills = set(task.get("required_skills", []))
        return required_skills.issubset(set(self.required_skills))
    
    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """Update role performance metrics."""
        self.metrics["tasks_completed"] += 1
        
        # Update success rate
        success = task_result.get("status") == "completed"
        total_tasks = self.metrics["tasks_completed"]
        self.metrics["success_rate"] = (
            (self.metrics["success_rate"] * (total_tasks - 1) + int(success))
            / total_tasks
        )
        
        # Update completion time
        if "completion_time" in task_result:
            avg_time = self.metrics["avg_completion_time"]
            new_time = task_result["completion_time"]
            self.metrics["avg_completion_time"] = (
                (avg_time * (total_tasks - 1) + new_time)
                / total_tasks
            )
    
    def get_agents_by_skill(self, skill: str) -> List[BaseAgent]:
        """Get all agents with a specific skill."""
        return [
            agent for agent in self.agents
            if skill in agent.skills
        ]
    
    def get_available_agents(self) -> List[BaseAgent]:
        """Get all agents currently available for tasks."""
        return [
            agent for agent in self.agents
            if not agent.is_busy
        ]
    
    def serialize(self) -> Dict[str, Any]:
        """Convert role to serializable format."""
        return {
            "name": self.name,
            "description": self.description,
            "required_skills": self.required_skills,
            "responsibilities": self.responsibilities,
            "parent_role": self.parent_role.name if self.parent_role else None,
            "metrics": self.metrics
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Role':
        """Create role from serialized data."""
        return cls(
            name=data["name"],
            description=data["description"],
            required_skills=data["required_skills"],
            responsibilities=data["responsibilities"]
        )

class RoleRegistry:
    """
    Global registry for role definitions.
    
    Features:
    - Role template management
    - Role hierarchy tracking
    - Role compatibility checking
    """
    
    _roles: Dict[str, Role] = {}
    
    @classmethod
    def register_role(cls, role: Role) -> None:
        """Register a new role template."""
        cls._roles[role.name] = role
    
    @classmethod
    def get_role(cls, name: str) -> Optional[Role]:
        """Get a role template by name."""
        return cls._roles.get(name)
    
    @classmethod
    def create_role(
        cls,
        name: str,
        description: str,
        **kwargs
    ) -> Role:
        """Create and register a new role."""
        role = Role(name, description, **kwargs)
        cls.register_role(role)
        return role
    
    @classmethod
    def get_compatible_roles(
        cls,
        agent: BaseAgent
    ) -> List[Role]:
        """Get all roles compatible with an agent's skills."""
        return [
            role for role in cls._roles.values()
            if set(role.required_skills).issubset(set(agent.skills))
        ]

# Register common roles
RoleRegistry.create_role(
    name="team_lead",
    description="Coordinates team activities and makes strategic decisions",
    required_skills=["leadership", "planning", "communication"],
    responsibilities=[
        "Task delegation",
        "Team coordination",
        "Strategy development",
        "Performance monitoring"
    ]
)

RoleRegistry.create_role(
    name="researcher",
    description="Gathers and analyzes information",
    required_skills=["research", "analysis", "documentation"],
    responsibilities=[
        "Information gathering",
        "Data analysis",
        "Report generation",
        "Knowledge management"
    ]
)

RoleRegistry.create_role(
    name="specialist",
    description="Provides domain-specific expertise",
    required_skills=["domain_expertise", "problem_solving"],
    responsibilities=[
        "Technical guidance",
        "Problem resolution",
        "Quality assurance",
        "Knowledge sharing"
    ]
)

RoleRegistry.create_role(
    name="coordinator",
    description="Facilitates communication and resource allocation",
    required_skills=["coordination", "communication", "organization"],
    responsibilities=[
        "Resource allocation",
        "Communication facilitation",
        "Progress tracking",
        "Issue resolution"
    ]
)

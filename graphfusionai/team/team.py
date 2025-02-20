"""
Core Team implementation for collaborative agent-based tasks.
"""
from typing import List, Dict, Any, Optional, Union
from graphfusionai.agents import BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.knowledge_graph import KnowledgeGraph
from .roles import Role, RoleRegistry
from .task import Task, TaskQueue
from .coordinator import TeamCoordinator

class Team:
    """
    A team of agents working together on tasks.
    
    Features:
    - Role-based agent organization
    - Task delegation and coordination
    - Shared knowledge and memory
    - Dynamic task prioritization
    - Real-time collaboration
    - Performance monitoring
    """
    
    def __init__(
        self,
        name: str,
        objective: str,
        coordinator: Optional[TeamCoordinator] = None,
        shared_memory: Optional[MemoryManager] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None
    ):
        """
        Initialize a new team.
        
        Args:
            name: Team name
            objective: Team's main objective
            coordinator: Optional custom coordinator
            shared_memory: Optional shared memory manager
            knowledge_graph: Optional shared knowledge graph
        """
        self.name = name
        self.objective = objective
        self.agents: Dict[str, BaseAgent] = {}
        self.roles: Dict[str, Role] = {}
        self.task_queue = TaskQueue()
        
        # Initialize components
        self.coordinator = coordinator or TeamCoordinator()
        self.shared_memory = shared_memory or MemoryManager()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0,
            "collaboration_score": 0
        }
    
    def add_agent(
        self,
        agent: BaseAgent,
        role: Union[str, Role],
        position: Optional[str] = None
    ) -> None:
        """
        Add an agent to the team with a specific role.
        
        Args:
            agent: Agent to add
            role: Role name or Role object
            position: Optional position within the role
        """
        # Get or create role
        if isinstance(role, str):
            role_obj = self.roles.get(role) or RoleRegistry.get_role(role)
            if not role_obj:
                raise ValueError(f"Role '{role}' not found")
        else:
            role_obj = role
            self.roles[role.name] = role_obj
            
        # Configure agent
        agent.set_memory_manager(self.shared_memory)
        agent.set_knowledge_graph(self.knowledge_graph)
        agent.set_role(role_obj, position)
        
        # Add to team
        self.agents[agent.id] = agent
        role_obj.add_agent(agent)
        
        # Notify coordinator
        self.coordinator.on_agent_added(agent, role_obj)
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the team."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            role = agent.role
            
            # Remove from role
            if role:
                role.remove_agent(agent)
                
            # Remove from team
            del self.agents[agent_id]
            
            # Notify coordinator
            self.coordinator.on_agent_removed(agent)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the team's queue."""
        self.task_queue.add(task)
        self.coordinator.on_task_added(task)
    
    def execute(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute team's tasks until completion or max steps reached.
        
        Args:
            max_steps: Optional maximum number of steps to execute
            
        Returns:
            Dictionary containing execution results and metrics
        """
        steps = 0
        results = []
        
        while (not self.task_queue.is_empty() and 
               (max_steps is None or steps < max_steps)):
            
            # Get next task
            task = self.task_queue.next()
            
            # Let coordinator assign task
            assigned_agents = self.coordinator.assign_task(task, self.agents)
            
            # Execute task
            try:
                task_result = self._execute_task(task, assigned_agents)
                results.append(task_result)
                self.metrics["tasks_completed"] += 1
                
            except Exception as e:
                self.metrics["tasks_failed"] += 1
                results.append({
                    "task": task,
                    "error": str(e),
                    "status": "failed"
                })
            
            steps += 1
            
        # Update metrics
        self._update_metrics()
        
        return {
            "results": results,
            "metrics": self.metrics,
            "steps": steps
        }
    
    def _execute_task(
        self,
        task: Task,
        assigned_agents: List[BaseAgent]
    ) -> Dict[str, Any]:
        """
        Execute a single task with assigned agents.
        
        Args:
            task: Task to execute
            assigned_agents: List of agents assigned to the task
            
        Returns:
            Dictionary containing task execution results
        """
        # Create task context
        context = {
            "team": self,
            "task": task,
            "assigned_agents": assigned_agents,
            "shared_memory": self.shared_memory,
            "knowledge_graph": self.knowledge_graph
        }
        
        # Execute task steps
        results = []
        for step in task.steps:
            step_result = self.coordinator.execute_step(
                step,
                assigned_agents,
                context
            )
            results.append(step_result)
            
            # Update task context
            context["previous_step"] = step
            context["previous_result"] = step_result
            
        return {
            "task": task,
            "results": results,
            "status": "completed"
        }
    
    def _update_metrics(self) -> None:
        """Update team performance metrics."""
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total_tasks > 0:
            self.metrics["success_rate"] = (
                self.metrics["tasks_completed"] / total_tasks
            )
            
        # Calculate collaboration score
        self.metrics["collaboration_score"] = self.coordinator.calculate_collaboration_score(
            self.agents,
            self.shared_memory,
            self.knowledge_graph
        )
    
    def get_agent_by_role(self, role_name: str) -> List[BaseAgent]:
        """Get all agents with a specific role."""
        if role_name in self.roles:
            return self.roles[role_name].agents
        return []
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by their ID."""
        return self.agents.get(agent_id)
    
    def broadcast_message(
        self,
        message: str,
        sender: Optional[BaseAgent] = None,
        recipients: Optional[List[BaseAgent]] = None
    ) -> None:
        """
        Broadcast a message to team members.
        
        Args:
            message: Message content
            sender: Optional sending agent
            recipients: Optional list of recipient agents (None for all)
        """
        target_agents = recipients or list(self.agents.values())
        self.coordinator.broadcast_message(message, sender, target_agents)
    
    def save_state(self, path: str) -> None:
        """
        Save team state to disk.
        
        Args:
            path: Path to save state
        """
        state = {
            "name": self.name,
            "objective": self.objective,
            "metrics": self.metrics,
            "memory": self.shared_memory.serialize(),
            "knowledge_graph": self.knowledge_graph.serialize(),
            "task_queue": self.task_queue.serialize()
        }
        import json
        with open(path, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load_state(cls, path: str) -> 'Team':
        """
        Load team state from disk.
        
        Args:
            path: Path to load state from
            
        Returns:
            Team instance with loaded state
        """
        import json
        with open(path, 'r') as f:
            state = json.load(f)
            
        team = cls(
            name=state["name"],
            objective=state["objective"]
        )
        
        team.metrics = state["metrics"]
        team.shared_memory.deserialize(state["memory"])
        team.knowledge_graph.deserialize(state["knowledge_graph"])
        team.task_queue.deserialize(state["task_queue"])
        
        return team

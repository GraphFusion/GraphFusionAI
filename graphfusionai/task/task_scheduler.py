"""
Task scheduling and agent assignment.
"""
from typing import Dict, Optional, List
from datetime import datetime
from .task import Task
from ..agents import BaseAgent

class TaskScheduler:
    """
    Schedules tasks and assigns agents.
    
    Features:
    - Agent skill matching
    - Load balancing
    - Performance-based assignment
    - Resource allocation
    """
    
    def __init__(self):
        """Initialize task scheduler."""
        self.assignments: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self.performance_history: Dict[str, List[float]] = {}  # agent_id -> success_rates
    
    def assign_agent(
        self,
        task: Task,
        agents: Dict[str, BaseAgent]
    ) -> Optional[BaseAgent]:
        """
        Find best agent for task.
        
        Args:
            task: Task to assign
            agents: Available agents
            
        Returns:
            Best agent or None if no suitable agent found
        """
        qualified_agents = []
        required_skills = set(task.required_skills)
        
        for agent in agents.values():
            if not self._is_agent_busy(agent):
                # Calculate agent score based on:
                # 1. Skill match
                # 2. Past performance
                # 3. Current load
                # 4. Resource availability
                score = self._calculate_agent_score(
                    agent,
                    required_skills,
                    task.resources
                )
                if score > 0:
                    qualified_agents.append((score, agent))
        
        if qualified_agents:
            # Get agent with highest score
            best_agent = max(qualified_agents, key=lambda x: x[0])[1]
            self._assign_task(best_agent.id, task.id)
            return best_agent
        
        return None
    
    def _is_agent_busy(self, agent: BaseAgent) -> bool:
        """Check if agent is busy."""
        return (
            agent.id in self.assignments and
            len(self.assignments[agent.id]) > 0
        )
    
    def _calculate_agent_score(
        self,
        agent: BaseAgent,
        required_skills: set,
        required_resources: Dict
    ) -> float:
        """Calculate agent's suitability score for task."""
        # Check if agent has all required skills
        agent_skills = set(agent.skills)
        if not required_skills.issubset(agent_skills):
            return 0.0
        
        # Calculate skill match score (0.4 weight)
        skill_match = len(required_skills & agent_skills) / len(required_skills)
        skill_score = 0.4 * skill_match
        
        # Calculate performance score (0.3 weight)
        performance_score = 0.3 * self._get_agent_performance(agent.id)
        
        # Calculate load score (0.2 weight)
        current_load = len(self.assignments.get(agent.id, []))
        load_score = 0.2 * (1.0 - (current_load / 5.0))  # Max 5 tasks
        
        # Calculate resource score (0.1 weight)
        resource_score = 0.1 * self._check_resource_availability(
            agent,
            required_resources
        )
        
        return skill_score + performance_score + load_score + resource_score
    
    def _get_agent_performance(self, agent_id: str) -> float:
        """Get agent's performance score."""
        history = self.performance_history.get(agent_id, [])
        if not history:
            return 0.5  # Default score for new agents
        return sum(history) / len(history)
    
    def _check_resource_availability(
        self,
        agent: BaseAgent,
        required_resources: Dict
    ) -> float:
        """Check if agent has required resources."""
        if not required_resources:
            return 1.0
        
        # Simple resource check - can be extended
        available_resources = getattr(agent, "resources", {})
        if not available_resources:
            return 0.0
        
        matched_resources = 0
        for resource, amount in required_resources.items():
            if (resource in available_resources and
                available_resources[resource] >= amount):
                matched_resources += 1
        
        return matched_resources / len(required_resources)
    
    def _assign_task(self, agent_id: str, task_id: str) -> None:
        """Record task assignment."""
        if agent_id not in self.assignments:
            self.assignments[agent_id] = []
        self.assignments[agent_id].append(task_id)
    
    def complete_task(
        self,
        agent_id: str,
        task_id: str,
        success: bool
    ) -> None:
        """
        Record task completion.
        
        Args:
            agent_id: ID of agent that completed task
            task_id: ID of completed task
            success: Whether task was successful
        """
        # Remove task assignment
        if agent_id in self.assignments:
            if task_id in self.assignments[agent_id]:
                self.assignments[agent_id].remove(task_id)
        
        # Update performance history
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        self.performance_history[agent_id].append(float(success))
        
        # Keep only last 10 performances
        if len(self.performance_history[agent_id]) > 10:
            self.performance_history[agent_id] = (
                self.performance_history[agent_id][-10:]
            )
    
    def get_agent_tasks(self, agent_id: str) -> List[str]:
        """Get all tasks assigned to agent."""
        return self.assignments.get(agent_id, [])

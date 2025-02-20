"""
Team coordination and task management.
"""
from typing import List, Dict, Any, Optional
from graphfusionai.agents import BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.knowledge_graph import KnowledgeGraph
from .task import Task

class TeamCoordinator:
    """
    Coordinates team activities and task execution.
    
    Features:
    - Task assignment
    - Resource allocation
    - Communication facilitation
    - Performance monitoring
    - Conflict resolution
    """
    
    def __init__(self, strategy: str = "balanced"):
        """
        Initialize coordinator.
        
        Args:
            strategy: Task assignment strategy
                     ("balanced", "specialized", "adaptive")
        """
        self.strategy = strategy
        self.task_history: List[Dict[str, Any]] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        
    def assign_task(
        self,
        task: Task,
        available_agents: Dict[str, BaseAgent]
    ) -> List[BaseAgent]:
        """
        Assign task to appropriate agents.
        
        Args:
            task: Task to assign
            available_agents: Dict of available agents
            
        Returns:
            List of assigned agents
        """
        if self.strategy == "balanced":
            return self._balanced_assignment(task, available_agents)
        elif self.strategy == "specialized":
            return self._specialized_assignment(task, available_agents)
        else:  # adaptive
            return self._adaptive_assignment(task, available_agents)
            
    def execute_step(
        self,
        step: Dict[str, Any],
        assigned_agents: List[BaseAgent],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a task step with assigned agents.
        
        Args:
            step: Step definition
            assigned_agents: Agents executing the step
            context: Execution context
            
        Returns:
            Step execution results
        """
        # Prepare step context
        step_context = {
            **context,
            "step": step,
            "coordinator": self
        }
        
        # Execute step with each agent
        results = []
        for agent in assigned_agents:
            try:
                result = agent.execute_step(step, step_context)
                results.append({
                    "agent": agent.id,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "agent": agent.id,
                    "status": "failed",
                    "error": str(e)
                })
                
        # Update statistics
        self._update_stats(step, assigned_agents, results)
        
        return {
            "step": step,
            "results": results,
            "status": "completed" if any(r["status"] == "success" for r in results) else "failed"
        }
        
    def broadcast_message(
        self,
        message: str,
        sender: Optional[BaseAgent],
        recipients: List[BaseAgent]
    ) -> None:
        """
        Broadcast message to team members.
        
        Args:
            message: Message content
            sender: Sending agent
            recipients: Recipient agents
        """
        for agent in recipients:
            try:
                agent.receive_message(message, sender)
            except Exception as e:
                print(f"Error sending message to {agent.id}: {e}")
                
    def calculate_collaboration_score(
        self,
        agents: Dict[str, BaseAgent],
        memory: MemoryManager,
        knowledge_graph: KnowledgeGraph
    ) -> float:
        """
        Calculate team collaboration effectiveness.
        
        Args:
            agents: Team agents
            memory: Shared memory
            knowledge_graph: Shared knowledge
            
        Returns:
            Collaboration score (0-1)
        """
        scores = []
        
        # Memory usage score
        memory_score = len(memory.get_memories()) / max(len(agents), 1)
        scores.append(min(memory_score, 1.0))
        
        # Knowledge sharing score
        knowledge_score = len(knowledge_graph.get_relationships()) / max(len(agents), 1)
        scores.append(min(knowledge_score, 1.0))
        
        # Task completion score
        if self.task_history:
            success_rate = sum(
                1 for task in self.task_history
                if task["status"] == "completed"
            ) / len(self.task_history)
            scores.append(success_rate)
            
        # Agent interaction score
        interaction_score = sum(
            len(agent.get_interactions())
            for agent in agents.values()
        ) / max(len(agents) ** 2, 1)
        scores.append(min(interaction_score, 1.0))
        
        return sum(scores) / len(scores) if scores else 0.0
        
    def _balanced_assignment(
        self,
        task: Task,
        available_agents: Dict[str, BaseAgent]
    ) -> List[BaseAgent]:
        """Assign task evenly among qualified agents."""
        qualified_agents = [
            agent for agent in available_agents.values()
            if set(task.required_skills).issubset(set(agent.skills))
            and not agent.is_busy
        ]
        
        # Sort by workload
        qualified_agents.sort(
            key=lambda a: len(self.agent_stats.get(a.id, {}).get("active_tasks", []))
        )
        
        # Select agents based on task complexity
        num_agents = min(
            len(qualified_agents),
            max(1, len(task.steps) // 2)  # 1 agent per 2 steps
        )
        
        return qualified_agents[:num_agents]
        
    def _specialized_assignment(
        self,
        task: Task,
        available_agents: Dict[str, BaseAgent]
    ) -> List[BaseAgent]:
        """Assign task to most specialized agents."""
        agent_scores = []
        
        for agent in available_agents.values():
            if agent.is_busy:
                continue
                
            # Calculate specialization score
            skill_match = len(
                set(task.required_skills) & set(agent.skills)
            ) / len(task.required_skills)
            
            # Consider past performance
            success_rate = self.agent_stats.get(agent.id, {}).get("success_rate", 0.5)
            
            # Combined score
            score = (skill_match * 0.7) + (success_rate * 0.3)
            agent_scores.append((score, agent))
            
        # Select top agents
        agent_scores.sort(reverse=True)
        return [
            agent for _, agent in agent_scores[:2]  # Top 2 specialists
            if agent_scores
        ]
        
    def _adaptive_assignment(
        self,
        task: Task,
        available_agents: Dict[str, BaseAgent]
    ) -> List[BaseAgent]:
        """Adaptively assign task based on current conditions."""
        if task.priority >= 8:  # High priority
            return self._specialized_assignment(task, available_agents)
        else:
            return self._balanced_assignment(task, available_agents)
            
    def _update_stats(
        self,
        step: Dict[str, Any],
        agents: List[BaseAgent],
        results: List[Dict[str, Any]]
    ) -> None:
        """Update agent and task statistics."""
        for result in results:
            agent_id = result["agent"]
            
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = {
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "active_tasks": [],
                    "success_rate": 0.0
                }
                
            stats = self.agent_stats[agent_id]
            
            if result["status"] == "success":
                stats["tasks_completed"] += 1
            else:
                stats["tasks_failed"] += 1
                
            total_tasks = stats["tasks_completed"] + stats["tasks_failed"]
            stats["success_rate"] = stats["tasks_completed"] / total_tasks
            
        # Update task history
        self.task_history.append({
            "step": step,
            "agents": [a.id for a in agents],
            "results": results,
            "timestamp": step.get("timestamp")
        })

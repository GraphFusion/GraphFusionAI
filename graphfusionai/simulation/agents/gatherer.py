"""
Gatherer agent implementation.
"""
from typing import Dict, Any, Tuple
import numpy as np
from ..agent import BaseAgent, AgentType, AgentState

class GathererAgent(BaseAgent):
    """Agent specialized in resource gathering."""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        """Initialize gatherer agent."""
        super().__init__(
            name=name,
            agent_type=AgentType.GATHERER,
            position=position,
            vision_range=3,
            movement_speed=1.0,
            communication_range=2
        )
        self.preferred_resources = set()
        self.resource_memory = {}
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """Enhanced decision making for gathering."""
        if self.stats.energy < 20:
            return AgentState.RESTING
            
        # Check for resources at current position
        for resource in perception['resources']:
            if resource.position == self.position:
                return AgentState.GATHERING
                
        # Update resource memory
        for resource in perception['resources']:
            self.resource_memory[resource.position] = {
                'type': resource.type,
                'quantity': resource.quantity,
                'last_seen': perception['step_count']
            }
            
        # Find nearest resource
        best_resource = None
        min_distance = float('inf')
        
        for pos, info in self.resource_memory.items():
            if info['last_seen'] < perception['step_count'] - 100:
                continue  # Resource info too old
                
            distance = np.sqrt(
                (pos[0] - self.position[0])**2 +
                (pos[1] - self.position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_resource = pos
                
        if best_resource:
            self.goals.append({
                'type': 'gather',
                'target': best_resource
            })
            return AgentState.MOVING
            
        # No resources found, explore
        return AgentState.EXPLORING
        
    def _gather_resources(self, environment: Any) -> None:
        """Enhanced resource gathering."""
        super()._gather_resources(environment)
        
        # Share resource location with nearby agents
        for agent in environment.communication_links.get(self, []):
            message = {
                'from': self.name,
                'type': 'resource_location',
                'content': self.resource_memory
            }
            agent.messages.append(message)

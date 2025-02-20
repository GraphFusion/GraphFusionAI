"""
Defender agent implementation.
"""
from typing import Dict, Any, Tuple
import numpy as np
from ..agent import BaseAgent, AgentType, AgentState

class DefenderAgent(BaseAgent):
    """Agent specialized in defending territory and resources."""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        """Initialize defender agent."""
        super().__init__(
            name=name,
            agent_type=AgentType.DEFENDER,
            position=position,
            vision_range=4,
            movement_speed=1.2,
            communication_range=2
        )
        self.patrol_points = []
        self.current_patrol_index = 0
        self.threat_memory = {}
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """Enhanced decision making for defense."""
        if self.stats.energy < 20:
            return AgentState.RESTING
            
        # Check for threats
        threats = self._assess_threats(perception)
        if threats:
            # Move to nearest threat
            nearest_threat = min(
                threats,
                key=lambda t: np.sqrt(
                    (t[0] - self.position[0])**2 +
                    (t[1] - self.position[1])**2
                )
            )
            
            self.goals = [{
                'type': 'defend',
                'target': nearest_threat
            }]
            return AgentState.MOVING
            
        # No threats, continue patrol
        if not self.patrol_points:
            self._setup_patrol(perception)
            
        if self.patrol_points:
            current_target = self.patrol_points[self.current_patrol_index]
            if self.position == current_target:
                # Move to next patrol point
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                current_target = self.patrol_points[self.current_patrol_index]
                
            self.goals = [{
                'type': 'patrol',
                'target': current_target
            }]
            return AgentState.MOVING
            
        return AgentState.EXPLORING
        
    def _assess_threats(self, perception: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Assess threats in the environment."""
        threats = []
        
        # Check for unknown agents near resources
        resource_positions = {
            r.position for r in perception['resources']
        }
        
        for agent in perception['agents']:
            if agent.name not in self.knowledge_base.get('agents', {}):
                # Unknown agent
                agent_pos = agent.position
                for resource_pos in resource_positions:
                    distance = np.sqrt(
                        (agent_pos[0] - resource_pos[0])**2 +
                        (agent_pos[1] - resource_pos[1])**2
                    )
                    if distance < 3:  # Threat if within 3 cells of resource
                        threats.append(agent_pos)
                        self.threat_memory[agent_pos] = {
                            'type': 'unknown_agent',
                            'step': perception['step_count']
                        }
                        
        return threats
        
    def _setup_patrol(self, perception: Dict[str, Any]) -> None:
        """Setup patrol points around resources."""
        resource_positions = {
            r.position for r in perception['resources']
        }
        
        if not resource_positions:
            # No resources to protect, patrol random points
            size = perception['size']
            self.patrol_points = [
                (np.random.randint(0, size[0]),
                 np.random.randint(0, size[1]))
                for _ in range(4)
            ]
            return
            
        # Create patrol points around resources
        for resource_pos in resource_positions:
            x, y = resource_pos
            # Add points in a diamond pattern around resource
            for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                patrol_x = x + dx
                patrol_y = y + dy
                if (0 <= patrol_x < perception['size'][0] and
                    0 <= patrol_y < perception['size'][1]):
                    self.patrol_points.append((patrol_x, patrol_y))
                    
    def _communicate(self, environment: Any) -> None:
        """Enhanced communication for defense."""
        super()._communicate(environment)
        
        # Share threat information
        for agent in environment.communication_links.get(self, []):
            message = {
                'from': self.name,
                'type': 'threat_alert',
                'content': self.threat_memory
            }
            agent.messages.append(message)

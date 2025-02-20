"""
Explorer agent implementation.
"""
from typing import Dict, Any, Tuple
import numpy as np
from ..agent import BaseAgent, AgentType, AgentState

class ExplorerAgent(BaseAgent):
    """Agent specialized in exploration."""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        """Initialize explorer agent."""
        super().__init__(
            name=name,
            agent_type=AgentType.EXPLORER,
            position=position,
            vision_range=5,  # Enhanced vision
            movement_speed=1.5,  # Faster movement
            communication_range=2
        )
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """Enhanced decision making for exploration."""
        if self.stats.energy < 20:
            return AgentState.RESTING
            
        # If current goal is exploration, continue
        if self.goals and self.goals[0]['type'] == 'explore':
            return AgentState.MOVING
            
        # Find unexplored areas
        known_terrain = self.knowledge_base.get('terrain', {})
        if len(known_terrain) < self.vision_range * self.vision_range:
            # Not enough terrain knowledge, explore randomly
            x = np.random.randint(0, perception['size'][0])
            y = np.random.randint(0, perception['size'][1])
            self.goals.append({
                'type': 'explore',
                'target': (x, y)
            })
            return AgentState.MOVING
            
        # Find areas with oldest exploration time
        exploration_times = {}
        for pos, info in known_terrain.items():
            if isinstance(info, dict) and 'last_seen' in info:
                exploration_times[pos] = info['last_seen']
                
        if exploration_times:
            # Find oldest explored area
            oldest_pos = min(exploration_times.items(), key=lambda x: x[1])[0]
            self.goals.append({
                'type': 'explore',
                'target': oldest_pos
            })
            return AgentState.MOVING
            
        return AgentState.EXPLORING

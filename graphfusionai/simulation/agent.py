"""
Base agent implementation for simulation.
"""
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

class AgentState(Enum):
    """Agent states."""
    IDLE = "idle"
    MOVING = "moving"
    GATHERING = "gathering"
    EXPLORING = "exploring"
    COMMUNICATING = "communicating"
    RESTING = "resting"

class AgentType(Enum):
    """Agent types."""
    EXPLORER = "explorer"
    GATHERER = "gatherer"
    COMMUNICATOR = "communicator"
    DEFENDER = "defender"
    LEADER = "leader"

@dataclass
class AgentStats:
    """Agent statistics."""
    health: float = 100.0
    energy: float = 100.0
    experience: float = 0.0
    level: int = 1
    resources: Dict[str, float] = field(default_factory=dict)
    skills: Dict[str, float] = field(default_factory=dict)

class BaseAgent:
    """Base agent class for simulation."""

    def __init__(self, 
                 name: str,
                 agent_type: AgentType,
                 position: Tuple[int, int],
                 vision_range: int = 3,
                 movement_speed: float = 1.0,
                 communication_range: int = 2):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            agent_type: Type of agent
            position: Initial position (x, y)
            vision_range: How far agent can see
            movement_speed: Movement speed in cells per step
            communication_range: Communication range in cells
        """
        self.name = name
        self.agent_type = agent_type
        self.position = position
        self.vision_range = vision_range
        self.movement_speed = movement_speed
        self.communication_range = communication_range
        
        # State
        self.state = AgentState.IDLE
        self.stats = AgentStats()
        self.memory: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.goals: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        
        # Initialize skills based on agent type
        self._initialize_skills()
        
    def _initialize_skills(self) -> None:
        """Initialize agent skills based on type."""
        base_skills = {
            "movement": 1.0,
            "perception": 1.0,
            "communication": 1.0,
            "resource_gathering": 1.0,
            "combat": 1.0
        }
        
        type_bonuses = {
            AgentType.EXPLORER: {"movement": 0.5, "perception": 0.5},
            AgentType.GATHERER: {"resource_gathering": 1.0},
            AgentType.COMMUNICATOR: {"communication": 1.0},
            AgentType.DEFENDER: {"combat": 1.0},
            AgentType.LEADER: {"communication": 0.5, "perception": 0.5}
        }
        
        # Apply base skills
        self.stats.skills.update(base_skills)
        
        # Apply type bonuses
        for skill, bonus in type_bonuses.get(self.agent_type, {}).items():
            self.stats.skills[skill] += bonus
            
    def perceive(self, environment: Any) -> Dict[str, Any]:
        """
        Perceive environment within vision range.
        
        Args:
            environment: Simulation environment
            
        Returns:
            Dictionary of perceived information
        """
        x, y = self.position
        vision_area = {
            'terrain': [],
            'resources': [],
            'agents': [],
            'events': []
        }
        
        # Get visible area bounds
        x_min = max(0, x - self.vision_range)
        x_max = min(environment.size[0], x + self.vision_range + 1)
        y_min = max(0, y - self.vision_range)
        y_max = min(environment.size[1], y + self.vision_range + 1)
        
        # Perceive terrain
        vision_area['terrain'] = [
            [environment.terrain[i][j]
             for j in range(y_min, y_max)]
            for i in range(x_min, x_max)
        ]
        
        # Perceive resources
        for resource in environment.resources:
            rx, ry = resource.position
            if x_min <= rx < x_max and y_min <= ry < y_max:
                vision_area['resources'].append(resource)
                
        # Perceive agents
        for agent in environment.agents:
            if agent != self:
                ax, ay = agent.position
                if x_min <= ax < x_max and y_min <= ay < y_max:
                    vision_area['agents'].append(agent)
                    
        # Get recent events in vision range
        vision_area['events'] = [
            event for event in environment.get_events()
            if 'position' in event and
            x_min <= event['position'][0] < x_max and
            y_min <= event['position'][1] < y_max
        ]
        
        return vision_area
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """
        Decide next action based on perception.
        
        Args:
            perception: Perceived information
            
        Returns:
            Next agent state
        """
        # Basic decision making - override in subclasses
        if not self.goals:
            return AgentState.IDLE
            
        current_goal = self.goals[0]
        
        if current_goal['type'] == 'gather':
            return AgentState.GATHERING
        elif current_goal['type'] == 'explore':
            return AgentState.EXPLORING
        elif current_goal['type'] == 'communicate':
            return AgentState.COMMUNICATING
        
        return AgentState.IDLE
        
    def act(self, environment: Any) -> None:
        """
        Execute action in environment.
        
        Args:
            environment: Simulation environment
        """
        # Perceive environment
        perception = self.perceive(environment)
        
        # Update knowledge
        self._update_knowledge(perception)
        
        # Decide next action
        self.state = self.decide(perception)
        
        # Execute action based on state
        if self.state == AgentState.MOVING:
            self._move(environment)
        elif self.state == AgentState.GATHERING:
            self._gather_resources(environment)
        elif self.state == AgentState.EXPLORING:
            self._explore(environment)
        elif self.state == AgentState.COMMUNICATING:
            self._communicate(environment)
        elif self.state == AgentState.RESTING:
            self._rest()
            
        # Update stats
        self._update_stats()
        
    def _update_knowledge(self, perception: Dict[str, Any]) -> None:
        """Update agent's knowledge base with new perception."""
        # Update terrain knowledge
        if 'terrain' in perception:
            self.knowledge_base.setdefault('terrain', {})
            for i, row in enumerate(perception['terrain']):
                for j, terrain in enumerate(row):
                    pos = (self.position[0] - self.vision_range + i,
                          self.position[1] - self.vision_range + j)
                    self.knowledge_base['terrain'][pos] = terrain
                    
        # Update resource knowledge
        if 'resources' in perception:
            self.knowledge_base.setdefault('resources', {})
            for resource in perception['resources']:
                self.knowledge_base['resources'][resource.position] = {
                    'type': resource.type,
                    'quantity': resource.quantity,
                    'last_seen': environment.step_count
                }
                
        # Update agent knowledge
        if 'agents' in perception:
            self.knowledge_base.setdefault('agents', {})
            for agent in perception['agents']:
                self.knowledge_base['agents'][agent.name] = {
                    'position': agent.position,
                    'type': agent.agent_type,
                    'last_seen': environment.step_count
                }
                
    def _move(self, environment: Any) -> None:
        """Move agent in environment."""
        if not self.goals or self.stats.energy < 10:
            return
            
        goal = self.goals[0]
        target = goal.get('target')
        if not target:
            return
            
        # Calculate direction to target
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance <= self.movement_speed:
            # Reached target
            self.position = target
            self.goals.pop(0)
        else:
            # Move toward target
            move_x = int(dx/distance * self.movement_speed)
            move_y = int(dy/distance * self.movement_speed)
            new_pos = (
                self.position[0] + move_x,
                self.position[1] + move_y
            )
            
            # Check if new position is valid
            if (0 <= new_pos[0] < environment.size[0] and
                0 <= new_pos[1] < environment.size[1] and
                environment.terrain[new_pos[0]][new_pos[1]] != TerrainType.OBSTACLE):
                self.position = new_pos
                self.stats.energy -= 5
                
    def _gather_resources(self, environment: Any) -> None:
        """Gather resources from current position."""
        if self.stats.energy < 20:
            return
            
        for resource in environment.resources:
            if resource.position == self.position:
                gather_amount = min(
                    10 * self.stats.skills['resource_gathering'],
                    resource.quantity
                )
                if gather_amount > 0:
                    resource.quantity -= gather_amount
                    self.stats.resources[resource.type] = (
                        self.stats.resources.get(resource.type, 0) + gather_amount
                    )
                    self.stats.energy -= 10
                    self.stats.experience += gather_amount
                    
    def _explore(self, environment: Any) -> None:
        """Explore environment."""
        if self.stats.energy < 15:
            return
            
        # Add unexplored areas to goals
        known_positions = set(self.knowledge_base.get('terrain', {}).keys())
        for x in range(max(0, self.position[0] - 5), 
                      min(environment.size[0], self.position[0] + 6)):
            for y in range(max(0, self.position[1] - 5),
                         min(environment.size[1], self.position[1] + 6)):
                if (x, y) not in known_positions:
                    self.goals.append({
                        'type': 'explore',
                        'target': (x, y)
                    })
                    
        self.stats.energy -= 5
        
    def _communicate(self, environment: Any) -> None:
        """Communicate with nearby agents."""
        if self.stats.energy < 10:
            return
            
        # Share knowledge with nearby agents
        for agent in environment.communication_links.get(self, []):
            message = {
                'from': self.name,
                'type': 'knowledge_share',
                'content': {
                    'resources': self.knowledge_base.get('resources', {}),
                    'terrain': self.knowledge_base.get('terrain', {})
                }
            }
            agent.messages.append(message)
            
        self.stats.energy -= 5
        
    def _rest(self) -> None:
        """Rest to recover energy."""
        self.stats.energy = min(100, self.stats.energy + 10)
        
    def _update_stats(self) -> None:
        """Update agent stats."""
        # Level up if enough experience
        while self.stats.experience >= 100 * self.stats.level:
            self.stats.experience -= 100 * self.stats.level
            self.stats.level += 1
            
            # Improve skills
            for skill in self.stats.skills:
                self.stats.skills[skill] += 0.1
                
        # Natural energy decay
        self.stats.energy = max(0, self.stats.energy - 1)
        
    def get_state(self) -> Dict[str, Any]:
        """Get agent state."""
        return {
            'name': self.name,
            'type': self.agent_type.value,
            'position': self.position,
            'state': self.state.value,
            'stats': {
                'health': self.stats.health,
                'energy': self.stats.energy,
                'experience': self.stats.experience,
                'level': self.stats.level,
                'resources': self.stats.resources.copy(),
                'skills': self.stats.skills.copy()
            }
        }

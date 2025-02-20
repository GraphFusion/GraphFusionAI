"""
Enhanced simulation environment with advanced features.
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

class TerrainType(Enum):
    """Types of terrain in the environment."""
    EMPTY = 0
    OBSTACLE = 1
    WATER = 2
    FOREST = 3
    MOUNTAIN = 4
    RESOURCE = 5

@dataclass
class Resource:
    """Resource in the environment."""
    type: str
    quantity: float
    position: Tuple[int, int]
    regeneration_rate: float = 0.0
    max_quantity: float = float('inf')

@dataclass
class SimulationState:
    """State of the simulation environment."""
    name: str
    size: Tuple[int, int]
    terrain: List[List[TerrainType]]
    resources: List[Resource]
    agents: List[Dict[str, Any]]
    timestamp: float
    step_count: int

class SimulationEnvironment:
    """Enhanced simulation environment with advanced features."""

    def __init__(self, name: str, size: Tuple[int, int] = (10, 10)):
        """
        Initialize simulation environment.
        
        Args:
            name: Environment name
            size: Grid size as (width, height)
        """
        self.name = name
        self.size = size
        self.terrain = [[TerrainType.EMPTY for _ in range(size[1])] for _ in range(size[0])]
        self.agents = []
        self.resources: List[Resource] = []
        self.step_count = 0
        self.start_time = time.time()
        self.paused = False
        self.events = []
        
        # Communication network
        self.communication_range = 3
        self.communication_links = {}
        
        # Performance metrics
        self.metrics = {
            'agent_counts': [],
            'resource_levels': [],
            'communication_density': [],
            'step_durations': []
        }

    def add_terrain(self, terrain_type: TerrainType, position: Tuple[int, int]) -> None:
        """Add terrain at specified position."""
        x, y = position
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.terrain[x][y] = terrain_type

    def add_resource(self, resource: Resource) -> None:
        """Add resource to environment."""
        self.resources.append(resource)

    def add_agent(self, agent: Any) -> None:
        """Add agent to environment."""
        self.agents.append(agent)
        self.update_agent_positions()
        self.update_communication_links()

    def remove_agent(self, agent: Any) -> None:
        """Remove agent from environment."""
        if agent in self.agents:
            self.agents.remove(agent)
            self.update_agent_positions()
            self.update_communication_links()

    def update_agent_positions(self) -> None:
        """Update agent positions and handle collisions."""
        position_map = {}
        for agent in self.agents:
            pos = tuple(agent.position)
            if pos in position_map:
                # Handle collision
                self.handle_collision(agent, position_map[pos])
            else:
                position_map[pos] = agent

    def handle_collision(self, agent1: Any, agent2: Any) -> None:
        """Handle collision between agents."""
        # Implement collision resolution logic
        self.events.append({
            'type': 'collision',
            'agents': [agent1.name, agent2.name],
            'position': agent1.position,
            'timestamp': time.time()
        })

    def update_communication_links(self) -> None:
        """Update communication links between agents."""
        self.communication_links.clear()
        for i, agent1 in enumerate(self.agents):
            self.communication_links[agent1] = []
            for j, agent2 in enumerate(self.agents):
                if i != j:
                    distance = np.sqrt(
                        (agent1.position[0] - agent2.position[0])**2 +
                        (agent1.position[1] - agent2.position[1])**2
                    )
                    if distance <= self.communication_range:
                        self.communication_links[agent1].append(agent2)

    def update_resources(self) -> None:
        """Update resource quantities."""
        for resource in self.resources:
            if resource.quantity < resource.max_quantity:
                resource.quantity = min(
                    resource.quantity + resource.regeneration_rate,
                    resource.max_quantity
                )

    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        return SimulationState(
            name=self.name,
            size=self.size,
            terrain=[[t.value for t in row] for row in self.terrain],
            resources=[asdict(r) for r in self.resources],
            agents=[{
                'name': a.name,
                'position': a.position,
                'state': getattr(a, 'state', None)
            } for a in self.agents],
            timestamp=time.time(),
            step_count=self.step_count
        )

    def save_state(self, filepath: str) -> None:
        """Save simulation state to file."""
        state = self.get_state()
        with open(filepath, 'w') as f:
            json.dump(asdict(state), f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load simulation state from file."""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
            state = SimulationState(**state_dict)
            
        # Restore state
        self.name = state.name
        self.size = state.size
        self.terrain = [[TerrainType(t) for t in row] for row in state.terrain]
        self.resources = [Resource(**r) for r in state.resources]
        self.step_count = state.step_count
        
        # Agents need special handling as they're complex objects
        self.events.append({
            'type': 'state_loaded',
            'timestamp': time.time(),
            'filepath': filepath
        })

    def step(self) -> None:
        """Execute one simulation step."""
        if self.paused:
            return
            
        start_time = time.time()
        
        # Update environment
        self.update_resources()
        self.update_communication_links()
        
        # Update agents
        for agent in self.agents:
            agent.act(self)
            
        # Update metrics
        self.metrics['agent_counts'].append(len(self.agents))
        self.metrics['resource_levels'].append(
            [r.quantity for r in self.resources]
        )
        self.metrics['communication_density'].append(
            sum(len(links) for links in self.communication_links.values()) / 
            (len(self.agents) * (len(self.agents) - 1)) if len(self.agents) > 1 else 0
        )
        self.metrics['step_durations'].append(time.time() - start_time)
        
        self.step_count += 1

    def pause(self) -> None:
        """Pause simulation."""
        self.paused = True
        self.events.append({
            'type': 'pause',
            'timestamp': time.time(),
            'step_count': self.step_count
        })

    def resume(self) -> None:
        """Resume simulation."""
        self.paused = False
        self.events.append({
            'type': 'resume',
            'timestamp': time.time(),
            'step_count': self.step_count
        })

    def get_metrics(self) -> Dict[str, List[Any]]:
        """Get simulation metrics."""
        return self.metrics.copy()

    def get_events(self, start_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get simulation events."""
        if start_time is None:
            return self.events.copy()
        return [e for e in self.events if e['timestamp'] >= start_time]

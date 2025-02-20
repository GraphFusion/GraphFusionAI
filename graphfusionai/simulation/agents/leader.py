"""
Leader agent implementation.
"""
from typing import Dict, Any, Tuple, List
import numpy as np
from ..agent import BaseAgent, AgentType, AgentState

class LeaderAgent(BaseAgent):
    """Agent specialized in coordinating other agents."""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        """Initialize leader agent."""
        super().__init__(
            name=name,
            agent_type=AgentType.LEADER,
            position=position,
            vision_range=4,
            movement_speed=1.0,
            communication_range=3
        )
        self.team = set()
        self.assignments = {}
        self.global_knowledge = {
            'resources': {},
            'terrain': {},
            'agents': {},
            'threats': {}
        }
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """Enhanced decision making for leadership."""
        if self.stats.energy < 20:
            return AgentState.RESTING
            
        # Process messages and update global knowledge
        self._process_messages()
        
        # Update team roster
        self._update_team(perception['agents'])
        
        # Make team assignments
        if not self.assignments or perception['step_count'] % 50 == 0:
            self._make_assignments(perception)
            
        # Find agent that hasn't been contacted recently
        agents_to_contact = self._find_agents_to_contact(perception)
        if agents_to_contact:
            nearest_agent = min(
                agents_to_contact,
                key=lambda a: np.sqrt(
                    (a.position[0] - self.position[0])**2 +
                    (a.position[1] - self.position[1])**2
                )
            )
            
            self.goals = [{
                'type': 'communicate',
                'target': nearest_agent.position
            }]
            return AgentState.MOVING
            
        # Move to central position among team
        if self.team:
            center = self._calculate_team_center()
            if center != self.position:
                self.goals = [{
                    'type': 'move',
                    'target': center
                }]
                return AgentState.MOVING
                
        return AgentState.COMMUNICATING
        
    def _process_messages(self) -> None:
        """Process received messages and update global knowledge."""
        for message in self.messages:
            if message['type'] == 'knowledge_share':
                for category, data in message['content'].items():
                    self.global_knowledge[category].update(data)
                    
            elif message['type'] == 'resource_location':
                self.global_knowledge['resources'].update(message['content'])
                
            elif message['type'] == 'threat_alert':
                self.global_knowledge['threats'].update(message['content'])
                
        self.messages.clear()
        
    def _update_team(self, visible_agents: List[Any]) -> None:
        """Update team roster."""
        # Add visible agents to team
        for agent in visible_agents:
            self.team.add(agent.name)
            self.global_knowledge['agents'][agent.name] = {
                'type': agent.agent_type,
                'position': agent.position,
                'stats': agent.stats
            }
            
    def _make_assignments(self, perception: Dict[str, Any]) -> None:
        """Make team assignments based on global knowledge."""
        self.assignments.clear()
        
        # Group agents by type
        agents_by_type = {}
        for agent_name, info in self.global_knowledge['agents'].items():
            agent_type = info['type']
            agents_by_type.setdefault(agent_type, []).append(agent_name)
            
        # Assign explorers to unexplored areas
        if AgentType.EXPLORER in agents_by_type:
            known_terrain = self.global_knowledge['terrain']
            unexplored = []
            for x in range(perception['size'][0]):
                for y in range(perception['size'][1]):
                    if (x, y) not in known_terrain:
                        unexplored.append((x, y))
                        
            if unexplored:
                for agent_name in agents_by_type[AgentType.EXPLORER]:
                    if unexplored:
                        target = unexplored.pop()
                        self.assignments[agent_name] = {
                            'type': 'explore',
                            'target': target
                        }
                        
        # Assign gatherers to resources
        if AgentType.GATHERER in agents_by_type:
            resources = list(self.global_knowledge['resources'].items())
            if resources:
                for agent_name in agents_by_type[AgentType.GATHERER]:
                    if resources:
                        resource_pos, info = resources.pop()
                        self.assignments[agent_name] = {
                            'type': 'gather',
                            'target': resource_pos
                        }
                        
        # Assign defenders to threats and resources
        if AgentType.DEFENDER in agents_by_type:
            threats = list(self.global_knowledge['threats'].items())
            resources = list(self.global_knowledge['resources'].items())
            
            for agent_name in agents_by_type[AgentType.DEFENDER]:
                if threats:
                    threat_pos, _ = threats.pop()
                    self.assignments[agent_name] = {
                        'type': 'defend',
                        'target': threat_pos
                    }
                elif resources:
                    resource_pos, _ = resources.pop()
                    self.assignments[agent_name] = {
                        'type': 'patrol',
                        'target': resource_pos
                    }
                    
        # Assign communicators to maintain network
        if AgentType.COMMUNICATOR in agents_by_type:
            team_positions = [
                info['position']
                for info in self.global_knowledge['agents'].values()
            ]
            
            if team_positions:
                for agent_name in agents_by_type[AgentType.COMMUNICATOR]:
                    target = self._find_communication_gap(team_positions)
                    if target:
                        self.assignments[agent_name] = {
                            'type': 'communicate',
                            'target': target
                        }
                        
    def _find_communication_gap(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Find position that would improve team communication."""
        if len(positions) < 2:
            return None
            
        # Find the two agents that are furthest apart
        max_distance = 0
        gap_pos = None
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distance = np.sqrt(
                    (pos1[0] - pos2[0])**2 +
                    (pos1[1] - pos2[1])**2
                )
                if distance > max_distance:
                    max_distance = distance
                    # Position halfway between the agents
                    gap_pos = (
                        int((pos1[0] + pos2[0]) / 2),
                        int((pos1[1] + pos2[1]) / 2)
                    )
                    
        return gap_pos
        
    def _find_agents_to_contact(self, perception: Dict[str, Any]) -> List[Any]:
        """Find agents that need to be contacted."""
        current_step = perception['step_count']
        agents_to_contact = []
        
        for agent in perception['agents']:
            if agent.name in self.assignments:
                # Check if agent needs new assignment
                last_update = self.global_knowledge['agents'].get(
                    agent.name, {}
                ).get('last_update', 0)
                
                if current_step - last_update > 50:
                    agents_to_contact.append(agent)
                    
        return agents_to_contact
        
    def _calculate_team_center(self) -> Tuple[int, int]:
        """Calculate central position among team members."""
        positions = [
            info['position']
            for info in self.global_knowledge['agents'].values()
        ]
        
        if not positions:
            return self.position
            
        center_x = int(np.mean([p[0] for p in positions]))
        center_y = int(np.mean([p[1] for p in positions]))
        
        return (center_x, center_y)
        
    def _communicate(self, environment: Any) -> None:
        """Enhanced communication for leadership."""
        super()._communicate(environment)
        
        # Share assignments and global knowledge
        for agent in environment.communication_links.get(self, []):
            if agent.name in self.assignments:
                message = {
                    'from': self.name,
                    'type': 'assignment',
                    'content': {
                        'assignment': self.assignments[agent.name],
                        'knowledge': {
                            'resources': self.global_knowledge['resources'],
                            'threats': self.global_knowledge['threats']
                        }
                    }
                }
                agent.messages.append(message)

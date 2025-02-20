"""
Communicator agent implementation.
"""
from typing import Dict, Any, Tuple
import numpy as np
from ..agent import BaseAgent, AgentType, AgentState

class CommunicatorAgent(BaseAgent):
    """Agent specialized in communication and information sharing."""
    
    def __init__(self, name: str, position: Tuple[int, int]):
        """Initialize communicator agent."""
        super().__init__(
            name=name,
            agent_type=AgentType.COMMUNICATOR,
            position=position,
            vision_range=3,
            movement_speed=1.0,
            communication_range=4  # Enhanced communication range
        )
        self.message_history = []
        self.agent_last_contact = {}
        
    def decide(self, perception: Dict[str, Any]) -> AgentState:
        """Enhanced decision making for communication."""
        if self.stats.energy < 20:
            return AgentState.RESTING
            
        # Process received messages
        self._process_messages()
        
        # Find agents that haven't been contacted recently
        current_step = perception['step_count']
        agents_to_contact = []
        
        for agent in perception['agents']:
            last_contact = self.agent_last_contact.get(agent.name, 0)
            if current_step - last_contact > 50:  # Contact every 50 steps
                agents_to_contact.append(agent)
                
        if agents_to_contact:
            # Move toward nearest agent
            nearest_agent = min(
                agents_to_contact,
                key=lambda a: np.sqrt(
                    (a.position[0] - self.position[0])**2 +
                    (a.position[1] - self.position[1])**2
                )
            )
            
            self.goals.append({
                'type': 'communicate',
                'target': nearest_agent.position
            })
            return AgentState.MOVING
            
        # If no agents to contact, explore
        return AgentState.EXPLORING
        
    def _process_messages(self) -> None:
        """Process received messages."""
        for message in self.messages:
            self.message_history.append(message)
            
            if message['type'] == 'knowledge_share':
                # Update knowledge base with shared information
                for category, data in message['content'].items():
                    self.knowledge_base.setdefault(category, {}).update(data)
                    
            elif message['type'] == 'resource_location':
                # Update resource knowledge
                self.knowledge_base.setdefault('resources', {}).update(
                    message['content']
                )
                
        self.messages.clear()
        
    def _communicate(self, environment: Any) -> None:
        """Enhanced communication."""
        super()._communicate(environment)
        
        # Record contact time
        current_step = environment.step_count
        for agent in environment.communication_links.get(self, []):
            self.agent_last_contact[agent.name] = current_step
            
        # Share aggregated knowledge
        for agent in environment.communication_links.get(self, []):
            message = {
                'from': self.name,
                'type': 'knowledge_share',
                'content': {
                    'resources': self.knowledge_base.get('resources', {}),
                    'terrain': self.knowledge_base.get('terrain', {}),
                    'agents': self.knowledge_base.get('agents', {})
                }
            }
            agent.messages.append(message)

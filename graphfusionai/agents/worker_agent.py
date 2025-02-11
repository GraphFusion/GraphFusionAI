from .base_agent import BaseAgent

class WorkerAgent(BaseAgent):
    def __init__(self, 
                 name: str, 
                 graph_network, 
                 knowledge_graph, 
                 llm_provider: str, 
                 api_key: str, 
                 model: str, 
                 memory_manager, 
                 action_dim: int):
        """
        Initializes the WorkerAgent, which extends BaseAgent.
        """
        super().__init__(name, graph_network, knowledge_graph, llm_provider, api_key, model, memory_manager)  
        
        self.action_dim = action_dim

    def process_input(self, input_data: str) -> None:
        """
        Processes specific tasks using the provided data.
        """
        print(f"[Worker {self.name}] Processing input: {input_data}")
    
    def decide(self, input_data: str) -> str:
        """
        Simple decision-making for task-specific actions.
        """
        print(f"[Worker {self.name}] Deciding next action for: {input_data}")
        return f"Action based on {input_data}"
    
    def communicate(self, other_agent: "BaseAgent", message: str) -> None:
        """
        Communicates with another agent.
        """
        print(f"[Worker {self.name}] Sending message to {other_agent.name}: {message}")
    
    def complete_task(self):
        """
        Marks the worker as available for a new task after completing the current one.
        """
        self.is_available = True
        print(f"[{self.name}] Task completed and ready for new task!")

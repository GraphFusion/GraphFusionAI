import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from graphfusionai.agents.worker_agent import WorkerAgent
from graphfusionai.agents.manager_agent import ManagerAgent
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.memory.memory_manager import MemoryManager  # Ensure MemoryManager is imported
from graphfusionai.llm import create_llm  # Assuming a method for creating LLM client

def main():
    # Initializing dimensions
    input_dim = 64
    memory_dim = 128
    context_dim = 96
    action_dim = 32
    n_workers = 4

    # Create graph network
    graph = GraphNetwork(feature_dim=input_dim, hidden_dim=memory_dim)

    # LLM provider, API key, model, and memory manager
    llm_provider = "openai"  
    api_key = "your_api_key_here"  # Replace with your API key
    model = "gpt-3.5-turbo"  
    memory_manager = MemoryManager()  # Initialize memory manager

    # Create manager agent
    name = "manager"
    manager = ManagerAgent(name, graph, None, llm_provider, api_key, model, memory_manager, n_workers)

    # Create worker agents
    workers = [
        WorkerAgent(f"worker_{i}", graph, None, llm_provider, api_key, model, memory_manager, action_dim)
        for i in range(n_workers)
    ]

    # Add agents to the graph
    graph.add_node("manager", {"type": "manager"})
    for i, worker in enumerate(workers):
        graph.add_node(f"worker_{i}", {"type": "worker"})
        graph.add_edge("manager", f"worker_{i}", edge_type="manages")

    # Simulation loop
    for step in range(100):
        # Generate some observation (simulated input data)
        observation = torch.randn(input_dim)

        # Manager assigns tasks
        tasks = [torch.randn(context_dim) for _ in range(n_workers)]

        # Workers process tasks
        worker_responses = []
        for worker, task in zip(workers, tasks):
            worker.process_input(f"Task: {task}")
            action = worker.decide(f"Task: {task}")
            worker_responses.append(action)

        # Manager collects responses
        for response in worker_responses:
            print(f"[Manager] Received response: {response}")

if __name__ == "__main__":
    main()

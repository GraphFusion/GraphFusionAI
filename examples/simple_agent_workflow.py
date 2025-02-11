import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphfusionai.agents.worker_agent import WorkerAgent
from graphfusionai.agents.manager_agent import ManagerAgent
from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph  # Ensure this is used
from graphfusionai.memory.memory_manager import MemoryManager  
from graphfusionai.llm import create_llm  

def main():
    # Define dimensions
    entity_dim = 32
    relation_dim = 16
    input_dim = 64
    memory_dim = 128
    context_dim = 96
    action_dim = 32  
    n_workers = 4

    # Create graph network and knowledge graph
    graph = GraphNetwork(feature_dim=input_dim, hidden_dim=memory_dim)
    knowledge_graph = KnowledgeGraph(entity_dim=entity_dim, relation_dim=relation_dim)  # Ensure this is properly initialized

    # LLM provider, model, and memory manager
    llm_provider = "huggingface"
    model = "gpt2"
    memory_manager = MemoryManager(
        input_dim=input_dim, 
        memory_dim=memory_dim, 
        context_dim=context_dim
    )

    # Create worker agents
    workers = [
        WorkerAgent(
            name=f"worker_{i}", 
            graph_network=graph, 
            knowledge_graph=knowledge_graph,  # Pass knowledge graph
            llm_provider=llm_provider, 
            model=model, 
            memory_manager=memory_manager,  
            action_dim=action_dim
        ) for i in range(n_workers)
    ]

    # Create manager agent
    manager = ManagerAgent(
        name="manager", 
        graph_network=graph, 
        knowledge_graph=knowledge_graph,  # Pass knowledge graph
        llm_provider=llm_provider, 
        model=model, 
        memory_manager=memory_manager, 
        n_workers=n_workers, 
        workers=workers
    )

    # Add agents to the graph
    graph.add_node("manager", {"type": "manager"})
    for i, worker in enumerate(workers):
        graph.add_node(f"worker_{i}", {"type": "worker"})
        graph.add_edge("manager", f"worker_{i}", edge_type="manages")

    # Simulation loop
    try:
        for step in range(100):
            print(f"\n[Simulation Step: {step+1}]")

            # Generate random observation
            observation = torch.randn(input_dim)

            # Manager assigns tasks
            tasks = [torch.randn(context_dim) for _ in range(n_workers)]

            # Workers process tasks and respond
            worker_responses = []
            for worker, task in zip(workers, tasks):
                worker.process_input(task)  # Pass tensor instead of a string
                action = worker.decide(task)  # Ensure `decide()` handles tensor input
                worker_responses.append(action)

            # Manager collects responses
            for response in worker_responses:
                print(f"[Manager] Received response: {response}")

    except Exception as e:
        print(f"[Error] Simulation encountered an issue: {e}")

if __name__ == "__main__":
    main()

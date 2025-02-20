"""
Basic example demonstrating core GraphFusionAI functionality.
"""
from graphfusionai import GraphNetwork, BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.tools.standard import WebSearchTool

def main():
    """Run basic agent example."""
    print("Initializing GraphFusionAI components...")
    
    # Initialize core components
    graph = GraphNetwork(feature_dim=768, hidden_dim=512)
    memory_manager = MemoryManager()
    llm_client = LiteLLMClient()
    
    # Create a search tool
    search_tool = WebSearchTool(
        api_key="your-tavily-api-key",  # Replace with your API key
        max_results=3,
        search_depth="basic"
    )
    
    # Create an agent
    agent = BaseAgent(
        role="Research Assistant",
        goal="Help users find and analyze information",
        backstory="I am an AI research assistant specializing in finding and analyzing information from various sources."
    )
    
    # Configure agent
    agent.set_memory_manager(memory_manager)
    agent.set_llm_client(llm_client)
    agent.add_tool(search_tool)
    
    print("\nAgent Configuration:")
    print(f"Role: {agent.role}")
    print(f"Goal: {agent.goal}")
    print(f"Tools: {[tool.__class__.__name__ for tool in agent.tools]}")
    
    # Example task: Research a topic
    task = {
        "type": "research",
        "content": "What are the latest developments in quantum computing?"
    }
    
    print("\nExecuting research task...")
    try:
        result = agent.execute_task(task)
        print("\nTask Result:")
        print(result)
        
        # Show memory state
        memories = memory_manager.get_memories()
        print("\nAgent Memories:")
        for memory in memories:
            print(f"- {memory}")
            
    except Exception as e:
        print(f"\nError executing task: {e}")
        
if __name__ == "__main__":
    main()

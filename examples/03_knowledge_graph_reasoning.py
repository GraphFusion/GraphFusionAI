"""
Example demonstrating knowledge graph-based reasoning in GraphFusionAI.
"""
from graphfusionai import GraphNetwork, BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.knowledge_graph import KnowledgeGraph
from graphfusionai.tools.standard import WebSearchTool

class KnowledgeAgent(BaseAgent):
    """Agent that builds and reasons with knowledge graphs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_graph = None
        
    def set_knowledge_graph(self, knowledge_graph):
        """Set the knowledge graph for this agent."""
        self.knowledge_graph = knowledge_graph
        
    def execute_task(self, task, context=None, tools=None):
        if task["type"] == "learn":
            return self._learn_about_topic(task["content"])
        elif task["type"] == "query":
            return self._query_knowledge(task["content"])
        elif task["type"] == "explain":
            return self._explain_relationship(task["content"])
        return super().execute_task(task, context, tools)
        
    def _learn_about_topic(self, topic):
        """Learn about a topic and add to knowledge graph."""
        # Search for information
        search_results = self.use_tool("web_search", topic)
        
        # Extract entities and relationships using LLM
        prompt = f"""
        From these search results about {topic}, identify:
        1. Key entities (concepts, people, technologies, etc.)
        2. Relationships between entities
        Format as: entity1 | relationship | entity2
        
        Search results: {search_results}
        """
        
        extraction = self.use_llm_for_query(prompt)
        
        # Add to knowledge graph
        for line in extraction.split('\n'):
            if '|' in line:
                entity1, relation, entity2 = [x.strip() for x in line.split('|')]
                self.knowledge_graph.add_relationship(entity1, relation, entity2)
                
        # Store in memory
        self.memory_manager.add_memory({
            "type": "knowledge",
            "content": f"Learned about {topic}",
            "entities": extraction
        })
        
        return f"Added knowledge about {topic} to graph"
        
    def _query_knowledge(self, query):
        """Query the knowledge graph."""
        # Find relevant entities and relationships
        entities = self.knowledge_graph.find_entities(query)
        relationships = self.knowledge_graph.get_relationships(entities)
        
        # Use LLM to formulate response
        prompt = f"""
        Based on these knowledge graph elements, answer the query: {query}
        
        Relevant information:
        Entities: {entities}
        Relationships: {relationships}
        """
        
        return self.use_llm_for_query(prompt)
        
    def _explain_relationship(self, entities):
        """Explain relationship between entities."""
        if isinstance(entities, str):
            # Parse entity pairs from string
            entity1, entity2 = [e.strip() for e in entities.split(',')]
        else:
            entity1, entity2 = entities
            
        # Find paths between entities
        paths = self.knowledge_graph.find_paths(entity1, entity2)
        
        # Use LLM to explain relationships
        prompt = f"""
        Explain the relationship between {entity1} and {entity2} based on these paths:
        
        {paths}
        """
        
        return self.use_llm_for_query(prompt)

def main():
    """Run knowledge graph reasoning example."""
    print("Initializing GraphFusionAI components...")
    
    # Initialize components
    graph = GraphNetwork(feature_dim=768, hidden_dim=512)
    knowledge_graph = KnowledgeGraph()
    memory_manager = MemoryManager()
    llm_client = LiteLLMClient()
    
    # Create search tool
    search_tool = WebSearchTool(
        api_key="your-tavily-api-key",  # Replace with your API key
        max_results=3,
        search_depth="basic"
    )
    
    # Create and configure agent
    agent = KnowledgeAgent(
        role="Knowledge Engineer",
        goal="Build and query knowledge graphs",
        backstory="I am a knowledge engineering specialist who builds and reasons with knowledge graphs."
    )
    
    agent.set_memory_manager(memory_manager)
    agent.set_llm_client(llm_client)
    agent.set_knowledge_graph(knowledge_graph)
    agent.add_tool(search_tool)
    
    print("\nAgent Configuration:")
    print(f"Role: {agent.role}")
    print(f"Goal: {agent.goal}")
    
    try:
        # Step 1: Learn about a topic
        print("\nLearning about quantum computing...")
        learn_task = {
            "type": "learn",
            "content": "quantum computing fundamentals and applications"
        }
        learn_result = agent.execute_task(learn_task)
        print(learn_result)
        
        # Step 2: Query knowledge
        print("\nQuerying knowledge...")
        query_task = {
            "type": "query",
            "content": "What are the main applications of quantum computing?"
        }
        query_result = agent.execute_task(query_task)
        print("\nQuery Result:")
        print(query_result)
        
        # Step 3: Explain relationships
        print("\nExplaining relationships...")
        explain_task = {
            "type": "explain",
            "content": "quantum computing, cryptography"
        }
        explain_result = agent.execute_task(explain_task)
        print("\nRelationship Explanation:")
        print(explain_result)
        
        # Show knowledge graph state
        print("\nKnowledge Graph State:")
        print(f"Entities: {knowledge_graph.get_entities()}")
        print(f"Relationships: {knowledge_graph.get_all_relationships()}")
        
    except Exception as e:
        print(f"\nError during knowledge operations: {e}")
        
if __name__ == "__main__":
    main()

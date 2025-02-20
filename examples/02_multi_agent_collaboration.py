"""
Example demonstrating multi-agent collaboration in GraphFusionAI.
"""
from graphfusionai import GraphNetwork, BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.tools.standard import WebSearchTool
from graphfusionai.knowledge_graph import KnowledgeGraph

class ResearchAgent(BaseAgent):
    """Agent specialized in research."""
    
    def execute_task(self, task, context=None, tools=None):
        if task["type"] == "research":
            # Use web search tool to gather information
            search_results = self.use_tool("web_search", task["content"])
            
            # Analyze results using LLM
            analysis = self.use_llm_for_query(
                f"Analyze these search results and provide key findings: {search_results}"
            )
            
            # Store findings in memory
            self.memory_manager.add_memory({
                "type": "research_findings",
                "content": analysis,
                "source": task["content"]
            })
            
            return analysis
        return super().execute_task(task, context, tools)

class AnalysisAgent(BaseAgent):
    """Agent specialized in analysis."""
    
    def execute_task(self, task, context=None, tools=None):
        if task["type"] == "analyze":
            # Get relevant memories
            memories = self.memory_manager.get_memories_by_type("research_findings")
            
            # Analyze findings using LLM
            analysis = self.use_llm_for_query(
                f"Synthesize these research findings into a comprehensive analysis: {memories}"
            )
            
            # Store analysis in memory
            self.memory_manager.add_memory({
                "type": "analysis",
                "content": analysis,
                "source": task["content"]
            })
            
            return analysis
        return super().execute_task(task, context, tools)

class SummaryAgent(BaseAgent):
    """Agent specialized in creating summaries."""
    
    def execute_task(self, task, context=None, tools=None):
        if task["type"] == "summarize":
            # Get relevant memories
            research = self.memory_manager.get_memories_by_type("research_findings")
            analysis = self.memory_manager.get_memories_by_type("analysis")
            
            # Create summary using LLM
            summary = self.use_llm_for_query(
                f"Create a concise summary combining research findings and analysis: {research} {analysis}"
            )
            
            # Store summary in memory
            self.memory_manager.add_memory({
                "type": "summary",
                "content": summary,
                "source": task["content"]
            })
            
            return summary
        return super().execute_task(task, context, tools)

def main():
    """Run multi-agent collaboration example."""
    print("Initializing GraphFusionAI components...")
    
    # Initialize shared components
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
    
    # Create agents
    researcher = ResearchAgent(
        role="Researcher",
        goal="Gather and analyze information",
        backstory="I am a research specialist focused on gathering accurate information."
    )
    
    analyst = AnalysisAgent(
        role="Analyst",
        goal="Analyze research findings",
        backstory="I am an analysis expert who synthesizes research findings."
    )
    
    summarizer = SummaryAgent(
        role="Summarizer",
        goal="Create concise summaries",
        backstory="I am a summary specialist who creates clear, concise summaries."
    )
    
    # Configure agents
    for agent in [researcher, analyst, summarizer]:
        agent.set_memory_manager(memory_manager)
        agent.set_llm_client(llm_client)
        
    researcher.add_tool(search_tool)
    
    print("\nAgent Configuration:")
    print("Researcher:", researcher.role)
    print("Analyst:", analyst.role)
    print("Summarizer:", summarizer.role)
    
    # Example: Research and analyze quantum computing
    topic = "recent breakthroughs in quantum computing"
    
    print(f"\nResearching topic: {topic}")
    
    try:
        # Step 1: Research
        research_task = {
            "type": "research",
            "content": f"What are the most significant {topic}?"
        }
        research_result = researcher.execute_task(research_task)
        print("\nResearch Findings:")
        print(research_result)
        
        # Step 2: Analysis
        analysis_task = {
            "type": "analyze",
            "content": topic
        }
        analysis_result = analyst.execute_task(analysis_task)
        print("\nAnalysis:")
        print(analysis_result)
        
        # Step 3: Summary
        summary_task = {
            "type": "summarize",
            "content": topic
        }
        summary_result = summarizer.execute_task(summary_task)
        print("\nFinal Summary:")
        print(summary_result)
        
        # Show memory state
        print("\nShared Memory State:")
        memories = memory_manager.get_memories()
        for memory in memories:
            print(f"\nType: {memory['type']}")
            print(f"Content: {memory['content'][:100]}...")
            
    except Exception as e:
        print(f"\nError during collaboration: {e}")
        
if __name__ == "__main__":
    main()

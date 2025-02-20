"""
Example of a research assistant agent that helps with information gathering and analysis.
"""
from graphfusionai import GraphNetwork, BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.knowledge_graph import KnowledgeGraph
from graphfusionai.tools.standard import WebSearchTool
import json
from datetime import datetime

class ResearchAssistant(BaseAgent):
    """Agent specialized in research and analysis."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.research_topics = {}
        self.current_topic = None
        
    def execute_task(self, task, context=None, tools=None):
        """Execute research-related tasks."""
        if task["type"] == "research":
            return self._conduct_research(task["content"])
        elif task["type"] == "analyze":
            return self._analyze_findings(task["content"])
        elif task["type"] == "summarize":
            return self._create_summary(task["content"])
        elif task["type"] == "fact_check":
            return self._fact_check(task["content"])
        return super().execute_task(task, context, tools)
        
    def _conduct_research(self, topic):
        """Conduct comprehensive research on a topic."""
        self.current_topic = topic
        
        # Initial search
        search_results = self.use_tool(
            "web_search",
            f"{topic} latest research findings"
        )
        
        # Extract key information using LLM
        prompt = f"""
        Analyze these search results about {topic}:
        
        {search_results}
        
        Provide:
        1. Key findings
        2. Important facts and statistics
        3. Recent developments
        4. Expert opinions
        5. Areas needing further research
        """
        
        analysis = self.use_llm_for_query(prompt)
        
        # Store research in memory
        self.memory_manager.add_memory({
            "type": "research",
            "topic": topic,
            "findings": analysis,
            "timestamp": str(datetime.now())
        })
        
        # Add to research topics
        self.research_topics[topic] = {
            "initial_findings": analysis,
            "analyses": [],
            "fact_checks": []
        }
        
        return analysis
        
    def _analyze_findings(self, research_data):
        """Analyze research findings in detail."""
        topic = research_data["topic"]
        findings = research_data["findings"]
        
        # Additional focused search
        focused_search = self.use_tool(
            "web_search",
            f"{topic} analysis methodology implications"
        )
        
        prompt = f"""
        Perform detailed analysis:
        
        Topic: {topic}
        Findings: {findings}
        Additional context: {focused_search}
        
        Provide:
        1. Methodology assessment
        2. Data quality evaluation
        3. Potential biases
        4. Implications of findings
        5. Recommendations for application
        """
        
        analysis = self.use_llm_for_query(prompt)
        
        # Store analysis in memory
        self.memory_manager.add_memory({
            "type": "analysis",
            "topic": topic,
            "analysis": analysis,
            "timestamp": str(datetime.now())
        })
        
        # Update research topics
        if topic in self.research_topics:
            self.research_topics[topic]["analyses"].append(analysis)
            
        return analysis
        
    def _create_summary(self, request):
        """Create a concise summary of research findings."""
        topic = request["topic"]
        audience = request.get("audience", "general")
        format_type = request.get("format", "detailed")
        
        # Get all relevant memories
        topic_memories = [
            m for m in self.memory_manager.get_memories()
            if m["type"] in ["research", "analysis"] and m["topic"] == topic
        ]
        
        prompt = f"""
        Create a {format_type} summary for {audience} audience:
        
        Topic: {topic}
        Research findings: {json.dumps([m["findings"] for m in topic_memories if "findings" in m])}
        Analyses: {json.dumps([m["analysis"] for m in topic_memories if "analysis" in m])}
        
        Focus on:
        1. Key takeaways
        2. Practical implications
        3. Actionable insights
        4. Future considerations
        """
        
        summary = self.use_llm_for_query(prompt)
        
        # Store summary in memory
        self.memory_manager.add_memory({
            "type": "summary",
            "topic": topic,
            "summary": summary,
            "audience": audience,
            "format": format_type,
            "timestamp": str(datetime.now())
        })
        
        return summary
        
    def _fact_check(self, statement):
        """Fact check a statement or finding."""
        # Search for verification
        search_results = self.use_tool(
            "web_search",
            f"fact check verify {statement}"
        )
        
        prompt = f"""
        Fact check this statement:
        
        Statement: {statement}
        Search results: {search_results}
        
        Provide:
        1. Verification status
        2. Supporting evidence
        3. Contradicting evidence
        4. Context and nuances
        5. Confidence level
        """
        
        fact_check = self.use_llm_for_query(prompt)
        
        # Store fact check in memory
        self.memory_manager.add_memory({
            "type": "fact_check",
            "statement": statement,
            "result": fact_check,
            "timestamp": str(datetime.now())
        })
        
        # Update research topics if relevant
        if self.current_topic:
            self.research_topics[self.current_topic]["fact_checks"].append({
                "statement": statement,
                "result": fact_check
            })
            
        return fact_check

def main():
    """Run research assistant example."""
    print("Initializing Research Assistant...")
    
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
    agent = ResearchAssistant(
        role="Research Assistant",
        goal="Conduct thorough research and analysis",
        backstory="I am an AI research assistant specializing in information gathering and analysis."
    )
    
    agent.set_memory_manager(memory_manager)
    agent.set_llm_client(llm_client)
    agent.add_tool(search_tool)
    
    print("\nResearch Assistant ready!")
    print(f"Role: {agent.role}")
    print(f"Goal: {agent.goal}")
    
    try:
        # Example 1: Conduct Research
        print("\n1. Conducting research...")
        research_task = {
            "type": "research",
            "content": "latest developments in quantum computing"
        }
        research_result = agent.execute_task(research_task)
        print("\nResearch Findings:")
        print(research_result)
        
        # Example 2: Analyze Findings
        print("\n2. Analyzing findings...")
        analysis_task = {
            "type": "analyze",
            "content": {
                "topic": "latest developments in quantum computing",
                "findings": research_result
            }
        }
        analysis_result = agent.execute_task(analysis_task)
        print("\nAnalysis Result:")
        print(analysis_result)
        
        # Example 3: Create Summary
        print("\n3. Creating summary...")
        summary_task = {
            "type": "summarize",
            "content": {
                "topic": "latest developments in quantum computing",
                "audience": "technical",
                "format": "detailed"
            }
        }
        summary_result = agent.execute_task(summary_task)
        print("\nSummary:")
        print(summary_result)
        
        # Example 4: Fact Check
        print("\n4. Fact checking...")
        fact_check_task = {
            "type": "fact_check",
            "content": "Quantum computers can break all current encryption methods"
        }
        fact_check_result = agent.execute_task(fact_check_task)
        print("\nFact Check Result:")
        print(fact_check_result)
        
        # Show research state
        print("\nResearch Topics State:")
        for topic, data in agent.research_topics.items():
            print(f"\nTopic: {topic}")
            print(f"Analyses: {len(data['analyses'])}")
            print(f"Fact Checks: {len(data['fact_checks'])}")
            
    except Exception as e:
        print(f"\nError during research: {e}")
        
if __name__ == "__main__":
    main()

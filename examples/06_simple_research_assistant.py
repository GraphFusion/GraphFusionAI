"""
Simplified example of a research assistant using LLM capabilities.
"""
from openai import OpenAI
import os
from typing import Dict, List, Any
import json
from datetime import datetime

class SimpleResearchAssistant:
    """A simplified research assistant that uses LLM for various research tasks."""
    
    def __init__(self, api_key: str = None):
        """Initialize the research assistant."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.memory = []
        self.current_topic = None
        
    def research_topic(self, topic: str) -> str:
        """Conduct research on a topic."""
        self.current_topic = topic
        prompt = f"""
        Please conduct research on this topic:
        {topic}

        Provide:
        1. Key findings and facts
        2. Main concepts
        3. Current developments
        4. Important debates or controversies
        5. Future implications

        Format as a clear research summary.
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("research", topic, response)
        return response
        
    def analyze_findings(self, research: str) -> str:
        """Analyze research findings."""
        prompt = f"""
        Please analyze these research findings:

        {research}

        Provide:
        1. Critical analysis
        2. Strengths and weaknesses
        3. Connections between concepts
        4. Implications
        5. Areas needing more research
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("analysis", research, response)
        return response
        
    def create_summary(self, content: str, audience: str = "general") -> str:
        """Create a summary for a specific audience."""
        prompt = f"""
        Create a summary of this content for a {audience} audience:

        {content}

        The summary should:
        1. Be appropriate for the audience level
        2. Focus on key points
        3. Use appropriate language and terminology
        4. Include relevant examples
        5. Highlight practical applications
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("summary", {"content": content, "audience": audience}, response)
        return response
        
    def fact_check(self, statement: str) -> str:
        """Fact check a statement."""
        prompt = f"""
        Please fact check this statement:

        "{statement}"

        Provide:
        1. Verification status
        2. Evidence supporting/contradicting
        3. Context needed
        4. Common misconceptions
        5. Confidence level in the assessment
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("fact_check", statement, response)
        return response
        
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"
            
    def _add_to_memory(self, task_type: str, input_data: Any, output: str):
        """Add interaction to memory."""
        self.memory.append({
            "type": task_type,
            "input": input_data,
            "output": output,
            "timestamp": str(datetime.now()),
            "topic": self.current_topic
        })
        
    def show_memory(self) -> List[Dict]:
        """Show all remembered interactions."""
        return self.memory
        
    def get_topic_memory(self, topic: str) -> List[Dict]:
        """Get all memory entries for a specific topic."""
        return [m for m in self.memory if m["topic"] == topic]

def main():
    """Run the research assistant example."""
    # Initialize assistant
    assistant = SimpleResearchAssistant()  # Make sure OPENAI_API_KEY is set in environment
    
    print(" Simple Research Assistant Demo")
    print("================================")
    
    try:
        # Example 1: Research Topic
        print("\n1. Research Topic Example")
        print("-----------------------")
        topic = "Recent advances in renewable energy storage"
        print(f"Researching topic: {topic}")
        research = assistant.research_topic(topic)
        print("\nResearch findings:")
        print(research)
        
        # Example 2: Analyze Findings
        print("\n2. Analysis Example")
        print("------------------")
        print("Analyzing the research findings...")
        analysis = assistant.analyze_findings(research)
        print("\nAnalysis:")
        print(analysis)
        
        # Example 3: Create Summaries
        print("\n3. Summary Creation Example")
        print("--------------------------")
        # Technical summary
        print("\nCreating technical summary...")
        tech_summary = assistant.create_summary(research + "\n" + analysis, "technical")
        print("\nTechnical Summary:")
        print(tech_summary)
        
        # General audience summary
        print("\nCreating general audience summary...")
        general_summary = assistant.create_summary(research + "\n" + analysis, "general")
        print("\nGeneral Audience Summary:")
        print(general_summary)
        
        # Example 4: Fact Checking
        print("\n4. Fact Checking Example")
        print("------------------------")
        statement = "Lithium-ion batteries are the only viable solution for grid-scale energy storage"
        print(f"Fact checking: '{statement}'")
        fact_check = assistant.fact_check(statement)
        print("\nFact Check Results:")
        print(fact_check)
        
        # Show topic memory
        print("\nMemory for Topic")
        print("----------------")
        topic_memory = assistant.get_topic_memory(topic)
        for i, mem in enumerate(topic_memory, 1):
            print(f"\nInteraction {i}:")
            print(f"Type: {mem['type']}")
            print(f"Timestamp: {mem['timestamp']}")
            print(f"Input: {str(mem['input'])[:100]}...")
            print(f"Output: {mem['output'][:100]}...")
            
    except Exception as e:
        print(f"\n Error: {str(e)}")
        
if __name__ == "__main__":
    main()

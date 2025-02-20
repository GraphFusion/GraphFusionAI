"""
Example of a coding assistant agent that helps with programming tasks.
"""
from graphfusionai import GraphNetwork, BaseAgent
from graphfusionai.memory import MemoryManager
from graphfusionai.llm import LiteLLMClient
from graphfusionai.knowledge_graph import KnowledgeGraph
from graphfusionai.tools.standard import WebSearchTool
import os
import tempfile

class CodingAssistant(BaseAgent):
    """Agent specialized in coding assistance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_memory = {}
        self.current_context = None
        
    def execute_task(self, task, context=None, tools=None):
        """Execute coding-related tasks."""
        if task["type"] == "code_review":
            return self._review_code(task["content"])
        elif task["type"] == "bug_fix":
            return self._fix_bug(task["content"])
        elif task["type"] == "code_generation":
            return self._generate_code(task["content"])
        elif task["type"] == "code_explanation":
            return self._explain_code(task["content"])
        return super().execute_task(task, context, tools)
        
    def _review_code(self, code):
        """Review code for improvements."""
        prompt = f"""
        Review this code and provide feedback on:
        1. Code quality and style
        2. Potential bugs or issues
        3. Performance considerations
        4. Security concerns
        5. Suggested improvements
        
        Code to review:
        ```python
        {code}
        ```
        """
        
        review = self.use_llm_for_query(prompt)
        
        # Store review in memory
        self.memory_manager.add_memory({
            "type": "code_review",
            "content": review,
            "code": code
        })
        
        return review
        
    def _fix_bug(self, bug_report):
        """Fix reported bug in code."""
        # Search for similar bugs
        search_results = self.use_tool(
            "web_search",
            f"how to fix {bug_report['error']} in {bug_report['language']}"
        )
        
        prompt = f"""
        Help fix this bug:
        
        Error: {bug_report['error']}
        Code:
        ```{bug_report['language']}
        {bug_report['code']}
        ```
        
        Context: {bug_report.get('context', 'No additional context')}
        Search results: {search_results}
        
        Provide:
        1. Root cause analysis
        2. Fixed code
        3. Explanation of the fix
        4. Prevention tips
        """
        
        solution = self.use_llm_for_query(prompt)
        
        # Store solution in memory
        self.memory_manager.add_memory({
            "type": "bug_fix",
            "error": bug_report['error'],
            "solution": solution
        })
        
        return solution
        
    def _generate_code(self, request):
        """Generate code based on requirements."""
        # Search for relevant examples
        search_results = self.use_tool(
            "web_search",
            f"code example {request['description']} in {request['language']}"
        )
        
        prompt = f"""
        Generate code based on these requirements:
        
        Description: {request['description']}
        Language: {request['language']}
        Requirements:
        {request['requirements']}
        
        Additional context:
        {request.get('context', 'No additional context')}
        
        Search results: {search_results}
        
        Provide:
        1. Complete, working code
        2. Explanation of the implementation
        3. Usage examples
        4. Any important notes or considerations
        """
        
        generated_code = self.use_llm_for_query(prompt)
        
        # Store in memory
        self.memory_manager.add_memory({
            "type": "generated_code",
            "description": request['description'],
            "code": generated_code
        })
        
        return generated_code
        
    def _explain_code(self, code_snippet):
        """Explain how code works."""
        prompt = f"""
        Explain this code in detail:
        
        ```python
        {code_snippet}
        ```
        
        Provide:
        1. High-level overview
        2. Line-by-line explanation
        3. Key concepts used
        4. Potential use cases
        5. Any important considerations
        """
        
        explanation = self.use_llm_for_query(prompt)
        
        # Store in memory
        self.memory_manager.add_memory({
            "type": "code_explanation",
            "code": code_snippet,
            "explanation": explanation
        })
        
        return explanation

def main():
    """Run coding assistant example."""
    print("Initializing Coding Assistant...")
    
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
    agent = CodingAssistant(
        role="Coding Assistant",
        goal="Help with programming tasks",
        backstory="I am an AI coding assistant with expertise in multiple programming languages."
    )
    
    agent.set_memory_manager(memory_manager)
    agent.set_llm_client(llm_client)
    agent.add_tool(search_tool)
    
    print("\nCoding Assistant ready!")
    print(f"Role: {agent.role}")
    print(f"Goal: {agent.goal}")
    
    try:
        # Example 1: Code Review
        print("\n1. Reviewing code...")
        code_to_review = """
        def fibonacci(n):
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            sequence = [0, 1]
            while len(sequence) < n:
                sequence.append(sequence[-1] + sequence[-2])
            return sequence
        """
        
        review_task = {
            "type": "code_review",
            "content": code_to_review
        }
        review_result = agent.execute_task(review_task)
        print("\nCode Review Result:")
        print(review_result)
        
        # Example 2: Bug Fix
        print("\n2. Fixing a bug...")
        bug_report = {
            "type": "bug_fix",
            "content": {
                "error": "IndexError: list index out of range",
                "language": "python",
                "code": """
                def get_last_elements(lst, n):
                    return lst[-n:]
                
                data = [1, 2, 3]
                result = get_last_elements(data, 5)
                """,
                "context": "Function should return last n elements of a list"
            }
        }
        fix_result = agent.execute_task(bug_report)
        print("\nBug Fix Result:")
        print(fix_result)
        
        # Example 3: Code Generation
        print("\n3. Generating code...")
        generation_task = {
            "type": "code_generation",
            "content": {
                "description": "Create a simple REST API endpoint",
                "language": "python",
                "requirements": """
                - Use FastAPI
                - Implement GET and POST methods
                - Include input validation
                - Add error handling
                - Include API documentation
                """,
                "context": "Part of a larger web service"
            }
        }
        generated_code = agent.execute_task(generation_task)
        print("\nGenerated Code:")
        print(generated_code)
        
        # Example 4: Code Explanation
        print("\n4. Explaining code...")
        explanation_task = {
            "type": "code_explanation",
            "content": """
            @contextmanager
            def timer():
                start = time.time()
                yield
                end = time.time()
                print(f"Execution time: {end - start:.2f} seconds")
            """
        }
        explanation = agent.execute_task(explanation_task)
        print("\nCode Explanation:")
        print(explanation)
        
        # Show memory state
        print("\nMemory State:")
        memories = memory_manager.get_memories()
        for memory in memories:
            print(f"\nType: {memory['type']}")
            print(f"Content: {memory['content'][:100]}...")
            
    except Exception as e:
        print(f"\nError during coding assistance: {e}")
        
if __name__ == "__main__":
    main()

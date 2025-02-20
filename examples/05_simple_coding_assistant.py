"""
Simplified example of a coding assistant using LLM capabilities.
"""
import openai
import os
from typing import Dict, List, Any
import json

class SimpleCodingAssistant:
    """A simplified coding assistant that uses LLM for various coding tasks."""
    
    def __init__(self, api_key: str = None):
        """Initialize the coding assistant."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.memory = []
        
    def review_code(self, code: str) -> str:
        """Review code and provide feedback."""
        prompt = f"""
        Please review this code and provide feedback on:
        1. Code quality and style
        2. Potential bugs
        3. Performance considerations
        4. Suggested improvements

        Code to review:
        ```python
        {code}
        ```
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("code_review", code, response)
        return response
        
    def fix_bug(self, code: str, error: str) -> str:
        """Fix a bug in the code."""
        prompt = f"""
        Please help fix this bug:

        Code with bug:
        ```python
        {code}
        ```

        Error message:
        {error}

        Please provide:
        1. What caused the bug
        2. How to fix it
        3. The corrected code
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("bug_fix", {"code": code, "error": error}, response)
        return response
        
    def generate_code(self, description: str) -> str:
        """Generate code based on description."""
        prompt = f"""
        Please write Python code that does the following:

        {description}

        Requirements:
        - Include necessary imports
        - Add clear comments
        - Follow PEP 8 style
        - Include basic error handling
        - Add a usage example
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("code_generation", description, response)
        return response
        
    def explain_code(self, code: str) -> str:
        """Explain how code works."""
        prompt = f"""
        Please explain this code in detail:

        ```python
        {code}
        ```

        Include:
        1. Overall purpose
        2. How it works
        3. Key components
        4. Any important Python concepts used
        """
        
        response = self._get_llm_response(prompt)
        self._add_to_memory("code_explanation", code, response)
        return response
        
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant."},
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
            "output": output
        })
        
    def show_memory(self) -> List[Dict]:
        """Show all remembered interactions."""
        return self.memory

def main():
    """Run the coding assistant example."""
    # Initialize assistant
    assistant = SimpleCodingAssistant()  # Make sure OPENAI_API_KEY is set in environment
    
    print("🤖 Simple Coding Assistant Demo")
    print("==============================")
    
    try:
        # Example 1: Code Review
        print("\n1. Code Review Example")
        print("----------------------")
        code_to_review = """
def factorial(n):
    if n == 0: return 1
    return n*factorial(n-1)
"""
        print("Code to review:")
        print(code_to_review)
        review = assistant.review_code(code_to_review)
        print("\nReview feedback:")
        print(review)
        
        # Example 2: Bug Fix
        print("\n2. Bug Fix Example")
        print("------------------")
        buggy_code = """
def get_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This will cause an error
result = get_average([])
"""
        print("Buggy code:")
        print(buggy_code)
        fix = assistant.fix_bug(buggy_code, "ZeroDivisionError: division by zero")
        print("\nBug fix suggestion:")
        print(fix)
        
        # Example 3: Code Generation
        print("\n3. Code Generation Example")
        print("--------------------------")
        description = """
Create a function that takes a list of strings and returns:
1. The longest string
2. Its length
3. Its position in the list
Handle empty lists appropriately.
"""
        print("Description:")
        print(description)
        generated_code = assistant.generate_code(description)
        print("\nGenerated code:")
        print(generated_code)
        
        # Example 4: Code Explanation
        print("\n4. Code Explanation Example")
        print("---------------------------")
        code_to_explain = """
@property
def full_name(self):
    return f"{self.first_name} {self.last_name}".strip()
"""
        print("Code to explain:")
        print(code_to_explain)
        explanation = assistant.explain_code(code_to_explain)
        print("\nExplanation:")
        print(explanation)
        
        # Show memory
        print("\nMemory of Interactions")
        print("---------------------")
        for i, mem in enumerate(assistant.show_memory(), 1):
            print(f"\nInteraction {i}:")
            print(f"Type: {mem['type']}")
            print(f"Input: {mem['input'][:100]}...")
            print(f"Output: {mem['output'][:100]}...")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
if __name__ == "__main__":
    main()

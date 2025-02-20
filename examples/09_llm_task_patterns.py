"""
Example demonstrating LLM task patterns in GraphFusionAI.
"""
from graphfusionai.task import (
    TaskManager,
    Task,
    TaskType,
    TaskStatus
)
from graphfusionai.task.llm_task import (
    LLMTask,
    LLMTaskType,
    create_completion_task,
    create_chat_task,
    create_code_task,
    create_analysis_task
)
from graphfusionai.agents import (
    CodingAgent,
    ResearchAgent,
    ReviewAgent
)
from graphfusionai.memory import MemoryManager
from graphfusionai.knowledge_graph import KnowledgeGraph

def main():
    """Run LLM task patterns demo."""
    print("🤖 LLM Task Patterns Demo")
    print("=========================\n")
    
    # Initialize components
    memory = MemoryManager()
    knowledge_graph = KnowledgeGraph()
    task_manager = TaskManager(memory, knowledge_graph)
    
    # Create agents
    agents = {
        "coder": CodingAgent(
            name="Senior Developer",
            skills=["coding", "system_design", "testing"]
        ),
        "researcher": ResearchAgent(
            name="Research Lead",
            skills=["research", "analysis", "documentation"]
        ),
        "reviewer": ReviewAgent(
            name="Code Reviewer",
            skills=["code_review", "best_practices", "security"]
        )
    }
    
    print("1. Creating LLM Tasks...")
    
    # 1. Code Generation Task
    code_task = create_code_task(
        prompt="""
        Create a Python function that:
        1. Takes a list of numbers
        2. Returns the sum of even numbers
        3. Uses list comprehension
        4. Includes type hints
        5. Has docstring with examples
        """,
        task_type=LLMTaskType.CODE_GENERATION,
        name="Generate Sum Even Function",
        provider="openai",
        model="gpt-4"
    )
    task_manager.add_task(code_task)
    print(f"\nTask: {code_task.name}")
    print(f"Type: {code_task.llm_type.value}")
    
    # 2. Code Review Task
    review_task = create_code_task(
        prompt="""
        def calculate_metrics(data):
            total = sum(data)
            avg = total / len(data)
            variance = sum((x - avg) ** 2 for x in data) / len(data)
            return {'total': total, 'average': avg, 'variance': variance}
        """,
        task_type=LLMTaskType.CODE_REVIEW,
        name="Review Metrics Function",
        provider="anthropic",
        model="claude-2",
        dependencies=[code_task.id]
    )
    task_manager.add_task(review_task)
    print(f"\nTask: {review_task.name}")
    print(f"Type: {review_task.llm_type.value}")
    
    # 3. Analysis Task with Chain of Thought
    analysis_task = create_analysis_task(
        data="""
        Performance Metrics:
        - Response Time: 245ms
        - Error Rate: 2.3%
        - CPU Usage: 78%
        - Memory Usage: 1.2GB
        - Throughput: 1200 req/s
        """,
        analysis_type="performance",
        name="Analyze System Performance",
        provider="openai",
        model="gpt-4",
        chain_of_thought=True
    )
    task_manager.add_task(analysis_task)
    print(f"\nTask: {analysis_task.name}")
    print(f"Type: {analysis_task.llm_type.value}")
    
    # 4. Chat Task for Planning
    chat_task = create_chat_task(
        messages=[
            {
                "role": "system",
                "content": "You are a technical lead planning system optimizations."
            },
            {
                "role": "user",
                "content": "Based on the performance analysis, what should we optimize first?"
            }
        ],
        name="Plan Optimizations",
        provider="openai",
        model="gpt-4",
        dependencies=[analysis_task.id]
    )
    task_manager.add_task(chat_task)
    print(f"\nTask: {chat_task.name}")
    print(f"Type: {chat_task.llm_type.value}")
    
    print("\n2. Executing Tasks...")
    results = task_manager.execute_tasks(agents)
    
    print("\n3. Execution Results:")
    print(f"Completed Tasks: {len(results['completed'])}")
    print(f"Failed Tasks: {len(results['failed'])}")
    print(f"Skipped Tasks: {len(results['skipped'])}")
    
    print("\nMetrics:")
    print(f"Total Time: {results['metrics']['avg_completion_time']:.2f} units")
    print(f"Success Rate: {results['metrics']['success_rate']*100:.2f}%")
    print(f"Max Parallel Tasks: {results['metrics']['parallel_tasks']}")
    
    print("\n4. Task Details:")
    for task_id in results['completed']:
        task = task_manager.get_task_by_id(task_id)
        print(f"\nTask: {task.name}")
        print(f"Status: {task.status.value}")
        print(f"Tokens: {task.metrics.get('token_count', 'N/A')}")
        print(f"Cost: ${task.metrics.get('total_cost', 0.0):.4f}")

if __name__ == "__main__":
    main()

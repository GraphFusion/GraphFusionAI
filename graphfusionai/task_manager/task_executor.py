import os
import sys
import time
import threading
from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graphfusionai.core.graph import GraphNetwork
from graphfusionai.core.knowledge_graph import KnowledgeGraph

class TaskExecutor:
    """Executes tasks with neural graph integration."""
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize the TaskExecutor.
        
        Args:
            max_retries (int): Maximum number of retry attempts for failed tasks
        """
        self.max_retries = max_retries
        self.lock = threading.Lock()

    def execute(self, agent: Any, task_name: str) -> bool:
        """
        Executes a task using the specified agent.
        
        Args:
            agent: The agent to execute the task
            task_name (str): Name of the task to execute
            
        Returns:
            bool: True if execution was successful, False otherwise
        """
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                print(f"🚀 Executing task '{task_name}' with agent {agent.name}")
                
                # Simulate task execution
                time.sleep(2)
                
                print(f"✅ Task '{task_name}' completed successfully")
                return True
                
            except Exception as e:
                retry_count += 1
                print(f"⚠️ Error executing task '{task_name}': {str(e)}")
                print(f"Retry attempt {retry_count}/{self.max_retries}")
                time.sleep(1)  # Wait before retrying
        
        print(f"❌ Failed to execute task '{task_name}' after {self.max_retries} attempts")
        return False

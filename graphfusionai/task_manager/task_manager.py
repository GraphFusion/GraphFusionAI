import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
import asyncio
from task_graph import TaskGraph
from task_executor import TaskExecutor
from scheduler import Scheduler
from agent_manager import AgentManager

logging.basicConfig(level=logging.INFO)

class TaskManager:
    """Graph-powered Task Manager for AI Agents."""

    def __init__(self):
        self.task_graph = TaskGraph()
        self.task_executor = TaskExecutor()
        self.scheduler = Scheduler(self.task_graph)
        self.agent_manager = AgentManager()

    def create_task(self, name, priority="medium", dependencies=None):
        """Creates a task and adds it to the task graph."""
        try:
            task = self.task_graph.add_task(name, priority, dependencies)
            logging.info(f"Task {name} created with priority {priority}")
            return task
        except Exception as e:
            logging.error(f"Failed to create task {name}: {e}")
            raise

    def assign_task(self, task, agent_name):
        """Assigns a task to an available agent."""
        try:
            agent = self.agent_manager.get_agent(agent_name)
            if agent:
                agent.assign(task)
                logging.info(f"Task {task} assigned to agent {agent_name}")
                return agent
            logging.warning(f"No available agent found for task {task}")
            return None
        except Exception as e:
            logging.error(f"Failed to assign task {task} to agent {agent_name}: {e}")
            raise

    async def execute_task(self, task):
        """Executes a task through an assigned agent."""
        try:
            result = await self.task_executor.run(task)
            logging.info(f"Task {task} executed successfully")
            return result
        except Exception as e:
            logging.error(f"Failed to execute task {task}: {e}")
            raise

    async def graph_powered_task(self, task_name, agent_name):
        """Unified function for graph-powered task execution."""
        try:
            task = self.create_task(task_name)
            agent = self.assign_task(task, agent_name)
            if agent:
                result = await self.execute_task(task)
                return result
            return f"No available agent for {task_name}"
        except Exception as e:
            logging.error(f"Error in graph-powered task {task_name}: {e}")
            raise

# Example usage
if __name__ == "__main__":
    task_manager = TaskManager()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task_manager.graph_powered_task("example_task", "agent_1"))
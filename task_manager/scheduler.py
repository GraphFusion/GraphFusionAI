import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import heapq
from collections import deque
from task_graph import TaskGraph
from agent_manager import AgentManager

class TaskScheduler:
    def __init__(self, agent_manager: AgentManager, task_graph: TaskGraph):
        self.agent_manager = agent_manager  # Manages available agents
        self.task_graph = task_graph  # Manages task dependencies
        self.task_queue = deque()  # Queue for tasks to be scheduled
        self.completed_tasks = set()  # Track completed tasks to resolve dependencies

    def schedule_task(self, task_id: str, priority: int = 1):
        """
        Add tasks to the task queue with their priority.
        The higher the priority number, the higher the priority.
        """
        heapq.heappush(self.task_queue, (-priority, task_id))  # Using max-heap based on priority

    def get_available_agent(self, required_skills: set) -> str:
        """
        Find the first available agent with the required skills.
        """
        for agent_id, agent in self.agent_manager.agents.items():
            if agent['status'] == 'Available' and required_skills.issubset(agent['skills']):
                return agent_id
        return None

    def can_task_run(self, task_id: str) -> bool:
        """
        Check if the task can run (i.e., its dependencies are complete).
        """
        dependencies = self.task_graph.get_task_dependencies(task_id)
        for dep in dependencies:
            if dep not in self.completed_tasks:
                return False  # If any dependency isn't completed, return False
        return True

    def assign_task_to_agent(self, task_id: str):
        """
        Assign the task to an available agent.
        """
        task = self.task_graph.get_task(task_id)
        required_skills = task['required_skills']
        agent_id = self.get_available_agent(required_skills)
        
        if agent_id:
            agent = self.agent_manager.agents[agent_id]
            agent['status'] = 'Busy'
            agent['task_count'] += 1
            print(f"Task {task_id} assigned to Agent {agent_id}.")
            return agent_id
        return None

    def mark_task_as_completed(self, task_id: str):
        """
        Mark a task as completed and update agent availability.
        """
        self.completed_tasks.add(task_id)
        print(f"Task {task_id} completed.")
        
        # Free up the agent after task completion
        for agent_id, agent in self.agent_manager.agents.items():
            if agent['task_count'] > 0:
                agent['task_count'] -= 1
                if agent['task_count'] == 0:
                    agent['status'] = 'Available'

    def run_scheduler(self):
        """
        Main scheduler loop to check tasks and assign them to agents.
        """
        while self.task_queue:
            # Get task with the highest priority
            _, task_id = heapq.heappop(self.task_queue)
            
            if self.can_task_run(task_id):
                # Assign task to an available agent if ready
                agent_id = self.assign_task_to_agent(task_id)
                if agent_id:
                    # Simulate task completion (in real scenario, you'd have more logic)
                    self.mark_task_as_completed(task_id)
            else:
                # If task cannot run, re-enqueue it for later execution
                heapq.heappush(self.task_queue, (_, task_id))

# Usage example:
if __name__ == '__main__':
    # Initialize TaskScheduler with the agent manager and task graph
    agent_manager = AgentManager()
    task_graph = TaskGraph()
    
    # Create tasks and assign priorities and dependencies
    task_graph.add_task('task1', {'required_skills': {'ML', 'Data Processing'}, 'dependencies': []})
    task_graph.add_task('task2', {'required_skills': {'Graph Analysis', 'NLP'}, 'dependencies': ['task1']})
    task_graph.add_task('task3', {'required_skills': {'Automation', 'Data Processing'}, 'dependencies': ['task1']})
    
    # Create and schedule tasks with priorities
    scheduler = TaskScheduler(agent_manager, task_graph)
    scheduler.schedule_task('task1', priority=2)
    scheduler.schedule_task('task2', priority=1)
    scheduler.schedule_task('task3', priority=3)
    
    # Run the scheduler to process tasks
    scheduler.run_scheduler()

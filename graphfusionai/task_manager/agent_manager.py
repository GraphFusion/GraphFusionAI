# graphfusionai/task_manager/agent_manager.py

class AgentManager:
    def __init__(self):
        """Initialize the agent manager with an empty registry of agents."""
        self.agents = {}  

    def register_agent(self, agent_id, skills):
        """
        Registers a new agent with a set of skills.
        :param agent_id: Unique identifier for the agent.
        :param skills: List of skills the agent is proficient in.
        """
        if agent_id in self.agents:
            print(f"Agent {agent_id} is already registered.")
            return

        self.agents[agent_id] = {
            "skills": set(skills),
            "status": "Available",  
            "task_count": 0
        }
        print(f"Agent {agent_id} registered with skills: {skills}")

    def update_agent_status(self, agent_id, status):
        """
        Updates an agent's availability status.
        :param agent_id: The agent to update.
        :param status: New status ('Available', 'Busy', 'Offline').
        """
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = status
            print(f"Agent {agent_id} is now {status}.")
        else:
            print(f"Agent {agent_id} not found.")

    def get_available_agents(self, required_skills):
        """
        Returns a list of agents that are available and have the required skills.
        :param required_skills: A set of skills required for the task.
        :return: List of suitable agents.
        """
        available_agents = [
            agent_id for agent_id, info in self.agents.items()
            if info["status"] == "Available" and required_skills.issubset(info["skills"])
        ]
        return available_agents

    def assign_task(self, task_id, required_skills):
        """
        Assigns a task to an available agent based on skill matching and load balancing.
        :param task_id: The ID of the task.
        :param required_skills: The set of skills required for the task.
        :return: Assigned agent ID or None if no suitable agent is found.
        """
        suitable_agents = self.get_available_agents(required_skills)
        if not suitable_agents:
            print(f"No available agents for task {task_id} requiring {required_skills}.")
            return None

        assigned_agent = min(suitable_agents, key=lambda a: self.agents[a]["task_count"])
        self.agents[assigned_agent]["status"] = "Busy"
        self.agents[assigned_agent]["task_count"] += 1

        print(f"Task {task_id} assigned to Agent {assigned_agent}.")
        return assigned_agent

    def release_agent(self, agent_id):
        """
        Marks an agent as available again after completing a task.
        :param agent_id: The ID of the agent to release.
        """
        if agent_id in self.agents:
            self.agents[agent_id]["task_count"] = max(0, self.agents[agent_id]["task_count"] - 1)
            if self.agents[agent_id]["task_count"] == 0:
                self.agents[agent_id]["status"] = "Available"
            print(f"Agent {agent_id} is now available.")
        else:
            print(f"Agent {agent_id} not found.")

    def list_agents(self):
        """Prints all registered agents and their statuses."""
        for agent_id, info in self.agents.items():
            print(f"Agent {agent_id}: {info}")


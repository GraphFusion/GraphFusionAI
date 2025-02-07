import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphfusionai.agents.base_agent import BaseAgent
from graphfusionai.core.knowledge_graph import KnowledgeGraph
from graphfusionai.llm import create_llm
from graphfusionai.core.graph import GraphNetwork
from typing import List, Dict

class SalesAgent(BaseAgent):
    def __init__(self, name: str, graph_network: GraphNetwork, knowledge_graph: KnowledgeGraph, llm_provider: str, api_key: str, model: str):
        super().__init__(name=name, 
                         graph_network=graph_network, 
                         knowledge_graph=knowledge_graph, 
                         llm_provider=llm_provider, 
                         api_key=api_key, 
                         model=model)

    def process_input(self, input_data: str) -> Dict[str, str]:
        """
        Process the input to extract meaningful data for lead identification, qualification, etc.
        """
        # Example: Input could be a customer query or profile data.
        response = self.use_llm_for_query(input_data)
        # Process response (e.g., extract key information about leads, opportunities, etc.)
        return {"response": response}

    def generate_lead(self, customer_data: Dict[str, str]) -> str:
        """
        Generate leads based on customer data. The agent will search through the knowledge graph to identify potential opportunities.
        """
        # Here we would have some logic to query the knowledge graph or external sources
        leads = self.knowledge_graph.query(customer_data)
        return leads

    def recommend_products(self, customer_data: Dict[str, str]) -> str:
        """
        Recommend products based on customer needs, analyzing past interactions, and knowledge graph data.
        """
        recommendations = self.knowledge_graph.query(customer_data, "product_recommendations")
        return recommendations

    def send_follow_up(self, customer_email: str, follow_up_message: str):
        """
        Automate email follow-ups to leads and prospects.
        """
        # Use LLM or direct communication tools to generate and send emails (e.g., email API)
        response = self.llm_client.call([{"role": "user", "content": follow_up_message}])
        print(f"Follow-up sent to {customer_email}: {response}")
        return response

    def update_sales_data(self, lead_id: str, status: str, interaction_notes: str):
        """
        Update sales status and interaction history in the knowledge graph.
        """
        self.knowledge_graph.add_relation(lead_id, "status", status)
        self.knowledge_graph.add_relation(lead_id, "interaction_notes", interaction_notes)
        print(f"Sales data for lead {lead_id} updated.")

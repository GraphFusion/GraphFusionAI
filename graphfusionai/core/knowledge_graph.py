import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple

class KnowledgeGraph(nn.Module):
    def __init__(self, entity_dim: int, relation_dim: int):
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        
        self.graph = nx.DiGraph()
        self.entities: Dict[str, torch.Tensor] = {}
        self.relations: Dict[str, torch.Tensor] = {}
        
        # Neural components
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, entity_dim)
        )
        
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim)
        )
        
        self.attention = nn.MultiheadAttention(entity_dim, num_heads=4)

    def add_entity(self, 
                   entity_id: str, 
                   features: torch.Tensor,
                   entity_type: Optional[str] = None) -> None:
        encoded_features = self.entity_encoder(features)
        self.entities[entity_id] = encoded_features
        self.graph.add_node(entity_id, type=entity_type)

    def add_relation(self,
                    from_entity: str,
                    to_entity: str,
                    relation_type: str,
                    features: Optional[torch.Tensor] = None) -> None:
        if features is not None:
            encoded_features = self.relation_encoder(features)
            self.relations[(from_entity, to_entity)] = encoded_features
        
        self.graph.add_edge(from_entity, to_entity, 
                           type=relation_type,
                           features=features)

    def query_subgraph(self, 
                      center_entity: str,
                      max_depth: int = 2) -> Tuple[nx.DiGraph, Dict[str, torch.Tensor]]:
        subgraph = nx.ego_graph(self.graph, center_entity, radius=max_depth)
         
        entity_embeddings = {
            node: self.entities[node]
            for node in subgraph.nodes()
            if node in self.entities
        }

        return subgraph, entity_embeddings

    def reason(self,
               query_entity: str,
               context_entities: List[str],) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform reasoning over the knowledge graph using attention."""
        if not context_entities:
            return self.entities[query_entity], None

        query = self.entities[query_entity].unsqueeze(0)
        context = torch.stark([self.entities[e] for e in context_entities])
        
        attended_context, attention_weights = self.attention(
            query_entity.unsqueeze(0),
            context.unsqueeze(0),
            context.unsqueeze(0)
        )

        return attended_context.squeeze(0), attention_weights

    def update_entity(self, 
                      entity_id: str, 
                      new_features: torch.Tensor) -> None:
        """Update the features of an entity with new information."""
        encoded_features = self.entity_encoder(new_features)
        self.entities[entity_id] = encoded_features

        
        

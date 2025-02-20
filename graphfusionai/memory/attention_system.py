import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

@dataclass
class AttentionContext:
    """Represents the current attention context"""
    focus_vector: torch.Tensor
    importance_weights: torch.Tensor
    temporal_weights: torch.Tensor
    context_window: List[str]
    active_tasks: List[str]

class MultiHeadFocusAttention(nn.Module):
    """
    Advanced attention mechanism with multiple focus heads for different aspects
    of memory relevance.
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Multi-head attention components
        self.query = nn.Linear(dim, num_heads * head_dim)
        self.key = nn.Linear(dim, num_heads * head_dim)
        self.value = nn.Linear(dim, num_heads * head_dim)
        
        # Output projection
        self.proj = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Focus mechanisms
        self.temporal_gate = nn.Linear(dim, 1)
        self.importance_gate = nn.Linear(dim, 1)
        self.relevance_gate = nn.Linear(dim, 1)
        
        # Context aggregation
        self.context_rnn = nn.GRU(dim, dim, batch_first=True)
        
    def forward(self, 
                query: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[AttentionContext] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply focused attention with multiple attention heads.
        
        Args:
            query: Query tensor [batch_size, query_len, dim]
            keys: Key tensor [batch_size, key_len, dim]
            values: Value tensor [batch_size, key_len, dim]
            mask: Optional attention mask
            context: Optional attention context
        """
        B, L, _ = query.shape
        
        # Project queries, keys, values
        q = self.query(query).view(B, L, self.num_heads, self.head_dim)
        k = self.key(keys).view(B, -1, self.num_heads, self.head_dim)
        v = self.value(values).view(B, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, key_len, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, key_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply focus mechanisms if context is provided
        if context is not None:
            # Temporal attention
            temporal_scores = self.temporal_gate(keys).squeeze(-1)
            temporal_weights = F.softmax(temporal_scores * context.temporal_weights, dim=-1)
            
            # Importance attention
            importance_scores = self.importance_gate(keys).squeeze(-1)
            importance_weights = F.softmax(importance_scores * context.importance_weights, dim=-1)
            
            # Relevance attention
            relevance_scores = self.relevance_gate(keys).squeeze(-1)
            relevance_weights = F.softmax(torch.matmul(keys, context.focus_vector.unsqueeze(-1)).squeeze(-1), dim=-1)
            
            # Combine attention mechanisms
            combined_weights = (temporal_weights + importance_weights + relevance_weights) / 3
            scores = scores * combined_weights.unsqueeze(1).unsqueeze(1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        output = torch.matmul(weights, v)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        
        # Project to output dimension
        output = self.proj(output)
        
        return output, weights

class AttentionSystem:
    """
    Advanced attention system for memory focus and retrieval.
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 context_size: int = 10):
        self.dim = dim
        self.context_size = context_size
        
        # Initialize attention mechanism
        self.attention = MultiHeadFocusAttention(dim, num_heads)
        
        # Initialize context
        self.reset_context()
        
        # Task-specific attention
        self.task_embeddings = nn.Parameter(torch.randn(100, dim))  # Support up to 100 task types
        self.task_attention = nn.MultiheadAttention(dim, num_heads)
        
    def reset_context(self):
        """Reset attention context"""
        self.context = AttentionContext(
            focus_vector=torch.zeros(self.dim),
            importance_weights=torch.ones(1),
            temporal_weights=torch.ones(1),
            context_window=[],
            active_tasks=[]
        )
        
    def update_focus(self, 
                    query_vector: torch.Tensor,
                    memory_vectors: torch.Tensor,
                    importance_scores: torch.Tensor,
                    timestamps: torch.Tensor) -> torch.Tensor:
        """
        Update attention focus based on current query and memory state.
        
        Args:
            query_vector: Current query embedding
            memory_vectors: Memory embeddings to attend to
            importance_scores: Importance scores for memories
            timestamps: Timestamp for each memory
        """
        # Update temporal weights
        current_time = datetime.now().timestamp()
        temporal_weights = torch.exp(-(current_time - timestamps) / 3600)  # Exponential decay
        self.context.temporal_weights = temporal_weights
        
        # Update importance weights
        self.context.importance_weights = importance_scores
        
        # Update focus vector using attention
        focus_vector, _ = self.attention(
            query_vector.unsqueeze(0).unsqueeze(0),
            memory_vectors.unsqueeze(0),
            memory_vectors.unsqueeze(0),
            context=self.context
        )
        self.context.focus_vector = focus_vector.squeeze()
        
        return self.context.focus_vector
        
    def apply_task_attention(self, memory_vectors: torch.Tensor, task_ids: List[int]) -> torch.Tensor:
        """Apply task-specific attention"""
        # Get task embeddings
        task_embeds = self.task_embeddings[task_ids]
        
        # Apply task attention
        task_context, _ = self.task_attention(
            task_embeds.unsqueeze(0),
            memory_vectors.unsqueeze(0),
            memory_vectors.unsqueeze(0)
        )
        
        return task_context.squeeze(0)
        
    def update_context_window(self, memory_key: str):
        """Update context window with recent memory access"""
        self.context.context_window.append(memory_key)
        if len(self.context.context_window) > self.context_size:
            self.context.context_window.pop(0)
            
    def get_focused_memories(self, 
                           memory_vectors: torch.Tensor,
                           query_vector: torch.Tensor,
                           top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get most relevant memories based on current focus and context.
        
        Args:
            memory_vectors: All memory vectors
            query_vector: Current query vector
            top_k: Number of memories to return
        """
        # Apply focused attention
        attended_memories, attention_weights = self.attention(
            query_vector.unsqueeze(0).unsqueeze(0),
            memory_vectors.unsqueeze(0),
            memory_vectors.unsqueeze(0),
            context=self.context
        )
        
        # Get top-k memories
        weights = attention_weights.squeeze()
        top_k_weights, top_k_indices = torch.topk(weights, k=top_k)
        top_k_memories = memory_vectors[top_k_indices]
        
        return top_k_memories, top_k_weights
        
    def analyze_attention(self) -> Dict[str, Any]:
        """Analyze current attention state"""
        return {
            "focus_strength": torch.norm(self.context.focus_vector).item(),
            "importance_distribution": {
                "mean": self.context.importance_weights.mean().item(),
                "std": self.context.importance_weights.std().item()
            },
            "temporal_distribution": {
                "mean": self.context.temporal_weights.mean().item(),
                "std": self.context.temporal_weights.std().item()
            },
            "context_window_size": len(self.context.context_window),
            "active_tasks": len(self.context.active_tasks)
        }

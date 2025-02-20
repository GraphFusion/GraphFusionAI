import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from collections import deque

class DynamicMemoryCell(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        memory_dim: int, 
        context_dim: int,
        lr: float = 0.001,
        temperature: float = 1.0,
        max_memories: Optional[int] = None,
        protection_threshold: float = 0.8,
        init_strategy: str = 'random',
        compression_ratio: float = 0.5,
        memory_growth_factor: float = 1.5
    ):
        super(DynamicMemoryCell, self).__init__()
        
        self.memory_dim = memory_dim
        self.temperature = temperature
        self.max_memories = max_memories
        self.protection_threshold = protection_threshold
        self.compression_ratio = compression_ratio
        self.memory_growth_factor = memory_growth_factor
        
        # Enhanced memory components
        self.keys = nn.Parameter(self._initialize_memory(memory_dim, input_dim, init_strategy))
        self.values = nn.Parameter(self._initialize_memory(memory_dim, context_dim, init_strategy))
        
        # Memory compression components
        self.compressor = nn.Sequential(
            nn.Linear(context_dim, int(context_dim * compression_ratio)),
            nn.ReLU(),
            nn.Linear(int(context_dim * compression_ratio), context_dim)
        )
        
        # Memory access tracking
        self.register_buffer('access_counts', torch.zeros(memory_dim))
        self.register_buffer('last_access_time', torch.zeros(memory_dim))
        self.register_buffer('protection_masks', torch.ones(memory_dim))
        self.register_buffer('memory_importance', torch.ones(memory_dim))
        
        # Enhanced attention mechanism
        self.key_attention = nn.MultiheadAttention(input_dim, num_heads=4)
        self.value_attention = nn.MultiheadAttention(context_dim, num_heads=4)
        
        self.update_layer = nn.GRUCell(context_dim, context_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.step_count = 0
        
        # Memory statistics
        self.access_history = deque(maxlen=1000)
        self.update_history = deque(maxlen=1000)

    def _initialize_memory(self, dim1: int, dim2: int, strategy: str) -> torch.Tensor:
        if strategy == 'zeros':
            return torch.zeros(dim1, dim2)
        elif strategy == 'uniform':
            return torch.rand(dim1, dim2) * 2 - 1
        elif strategy == 'orthogonal':
            return torch.nn.init.orthogonal_(torch.empty(dim1, dim2))
        else:  # 'random'
            return torch.randn(dim1, dim2)

    def forward(self, input_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enhanced attention mechanism with multi-head attention
        key_output, key_attention = self.key_attention(
            input_embedding.unsqueeze(0),
            self.keys.unsqueeze(0),
            self.keys.unsqueeze(0)
        )
        
        value_output, value_attention = self.value_attention(
            key_output,
            self.values.unsqueeze(0),
            self.values.unsqueeze(0)
        )
        
        retrieved_memory = value_output.squeeze(0)
        attention_scores = key_attention.squeeze(0)
        
        # Update access statistics
        self._update_access_stats(attention_scores)
        
        # Record access for analytics
        self.access_history.append({
            'step': self.step_count,
            'attention_scores': attention_scores.detach().mean().item()
        })
        
        return retrieved_memory, attention_scores

    def _update_access_stats(self, attention_scores: torch.Tensor) -> None:
        """Update memory access statistics"""
        self.access_counts += attention_scores.detach()
        self.last_access_time = torch.where(
            attention_scores.detach() > 0.1,
            torch.full_like(self.last_access_time, self.step_count),
            self.last_access_time
        )
        
        # Update importance based on access patterns
        recency_factor = torch.exp(-(self.step_count - self.last_access_time) / 1000)
        frequency_factor = torch.log1p(self.access_counts)
        self.memory_importance = 0.7 * recency_factor + 0.3 * frequency_factor
        
        self.step_count += 1

    def compress_memories(self) -> None:
        """Compress memory representations"""
        with torch.no_grad():
            compressed_values = self.compressor(self.values)
            self.values.data = compressed_values

    def expand_memory(self) -> None:
        """Dynamically expand memory capacity"""
        if self.max_memories is None or self.memory_dim < self.max_memories:
            new_size = int(self.memory_dim * self.memory_growth_factor)
            if self.max_memories:
                new_size = min(new_size, self.max_memories)
                
            self._resize_memories(new_size)

    def _resize_memories(self, new_size: int) -> None:
        """Resize memory matrices and buffers"""
        # Resize keys and values
        new_keys = torch.zeros(new_size, self.keys.size(1))
        new_values = torch.zeros(new_size, self.values.size(1))
        
        new_keys[:self.memory_dim] = self.keys.data
        new_values[:self.memory_dim] = self.values.data
        
        self.keys = nn.Parameter(new_keys)
        self.values = nn.Parameter(new_values)
        
        # Resize tracking buffers
        self.register_buffer('access_counts', torch.zeros(new_size))
        self.register_buffer('last_access_time', torch.zeros(new_size))
        self.register_buffer('protection_masks', torch.ones(new_size))
        self.register_buffer('memory_importance', torch.ones(new_size))
        
        self.memory_dim = new_size

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage and performance statistics"""
        return {
            'total_memories': self.memory_dim,
            'active_memories': (self.access_counts > 0).sum().item(),
            'protected_memories': (self.protection_masks > 0).sum().item(),
            'avg_importance': self.memory_importance.mean().item(),
            'memory_utilization': (self.access_counts > 0).float().mean().item(),
            'access_history': list(self.access_history),
            'update_history': list(self.update_history)
        }

    def update_memory(
        self, 
        input_embedding: torch.Tensor, 
        context: torch.Tensor,
        force_update: bool = False
    ) -> None:
        """
        Updates memory dynamically based on new input with protection mechanisms.

        Args:
            input_embedding (torch.Tensor): New input representation
            context (torch.Tensor): Context vector to update memory
            force_update (bool): Whether to force update protected memories
        """
        attention_scores = torch.softmax(self.attention(input_embedding) / self.temperature, dim=-1)
        updated_values = self.update_layer(context)
        
        # Calculate update gate
        gate_input = torch.cat([input_embedding, context], dim=-1)
        update_gate = self.gate(gate_input)
        
        # Apply protection mask unless forced
        if not force_update:
            attention_scores = attention_scores * self.protection_masks
        
        # Weight the update based on attention and gate
        memory_update = attention_scores.unsqueeze(-1) * updated_values.unsqueeze(0)
        alpha = update_gate.item()  # Dynamic interpolation factor
        new_values = (1 - alpha) * self.values + alpha * memory_update.sum(dim=0)
        
        loss = torch.nn.functional.mse_loss(self.values, new_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cleanup_memory(self, age_threshold: int = 1000) -> None:
        """
        Cleanup old or rarely accessed memories.
        
        Args:
            age_threshold (int): Number of steps before considering memory old
        """
        if self.max_memories is None:
            return
            
        current_time = self.step_count
        age = current_time - self.last_access_time
        
        # Calculate importance scores
        importance = self.access_counts / (age + 1)
        
        # Find least important unprotected memories
        unprotected_mask = ~(self.protection_masks.bool())
        unprotected_importance = importance * unprotected_mask
        
        # Reset least important memories
        _, indices = torch.topk(unprotected_importance, 
                              k=max(0, self.memory_dim - self.max_memories),
                              largest=False)
        
        with torch.no_grad():
            self.keys.data[indices] = self._initialize_memory(len(indices), self.keys.size(1), 'random')
            self.values.data[indices] = self._initialize_memory(len(indices), self.values.size(1), 'random')
            self.access_counts[indices] = 0
            self.last_access_time[indices] = current_time

    def protect_memory(self, attention_scores: torch.Tensor) -> None:
        """
        Protect frequently accessed memories.
        
        Args:
            attention_scores (torch.Tensor): Latest attention scores
        """
        # Update protection masks based on access frequency
        protection_scores = self.access_counts / (self.step_count + 1)
        self.protection_masks = (protection_scores > self.protection_threshold).float()

    def save_memory(self, path: str) -> None:
        """
        Saves the memory state.

        Args:
            path (str): File path to save the memory state.
        """
        torch.save({'keys': self.keys.data, 'values': self.values.data}, path)

    def load_memory(self, path: str) -> None:
        """
        Loads the memory state.

        Args:
            path (str): File path to load the memory state from.
        """
        checkpoint = torch.load(path)
        with torch.no_grad():
            self.keys.copy_(checkpoint['keys'])
            self.values.copy_(checkpoint['values'])

# Initialize with memory management
memory_cell = DynamicMemoryCell(
    input_dim=256,
    memory_dim=1000,
    context_dim=512,
    max_memories=800,  # Maintain only 800 most important memories
    protection_threshold=0.8,  # Protect memories accessed > 80% of average
    init_strategy='random',
    compression_ratio=0.5,
    memory_growth_factor=1.5
)

# Initialize step counter
current_step = 0

# Regular usage with protection
embedding = torch.randn(256)
context = torch.randn(512)
output, attention = memory_cell(embedding)
memory_cell.protect_memory(attention)
memory_cell.update_memory(embedding, context)

# Periodic cleanup (every 1000 steps)
if current_step % 1000 == 0:
    memory_cell.cleanup_memory(age_threshold=1000)

# Increment step counter
current_step += 1

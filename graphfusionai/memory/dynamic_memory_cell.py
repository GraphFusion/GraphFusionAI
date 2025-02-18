import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional, Tuple
import numpy as np

class DynamicMemoryCell(nn.Module):
    """
    Implements a dynamic memory cell that learns and updates stored representations
    using an attention-based mechanism with advanced memory management.
    """

    def __init__(
        self, 
        input_dim: int, 
        memory_dim: int, 
        context_dim: int, 
        lr: float = 0.001, 
        temperature: float = 1.0,
        max_memories: Optional[int] = None,
        protection_threshold: float = 0.8,
        init_strategy: str = 'random'
    ):
        """
        Args:
            input_dim (int): Dimensionality of input embeddings
            memory_dim (int): Number of memory slots
            context_dim (int): Dimensionality of context vectors
            lr (float): Learning rate for memory updates
            temperature (float): Scaling factor for attention softmax
            max_memories (Optional[int]): Maximum number of memory slots to maintain
            protection_threshold (float): Threshold for memory protection (0-1)
            init_strategy (str): Memory initialization strategy ('random', 'zeros', 'uniform')
        """
        super(DynamicMemoryCell, self).__init__()
        self.memory_dim = memory_dim
        self.temperature = temperature
        self.max_memories = max_memories
        self.protection_threshold = protection_threshold
        
        # Initialize memories based on strategy
        self.keys = nn.Parameter(self._initialize_memory(memory_dim, input_dim, init_strategy))
        self.values = nn.Parameter(self._initialize_memory(memory_dim, context_dim, init_strategy))
        
        # Usage tracking
        self.register_buffer('access_counts', torch.zeros(memory_dim))
        self.register_buffer('last_access_time', torch.zeros(memory_dim))
        self.register_buffer('protection_masks', torch.ones(memory_dim))
        
        self.attention = nn.Linear(input_dim, memory_dim)
        self.update_layer = nn.Linear(context_dim, context_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + context_dim, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.step_count = 0

    def _initialize_memory(self, dim1: int, dim2: int, strategy: str) -> torch.Tensor:
        """Initialize memory using the specified strategy."""
        if strategy == 'zeros':
            return torch.zeros(dim1, dim2)
        elif strategy == 'uniform':
            return torch.rand(dim1, dim2) * 2 - 1
        else:  # 'random'
            return torch.randn(dim1, dim2)

    def forward(self, input_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves relevant memory using an attention mechanism.

        Args:
            input_embedding (torch.Tensor): Input vector to query memory.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Retrieved context vector, attention scores)
        """
        attention_scores = torch.softmax(self.attention(input_embedding) / self.temperature, dim=-1)
        retrieved_memory = torch.matmul(attention_scores, self.values)
        
        # Update access statistics
        self.access_counts += attention_scores.detach()
        self.last_access_time = torch.where(
            attention_scores.detach() > 0.1,
            torch.full_like(self.last_access_time, self.step_count),
            self.last_access_time
        )
        self.step_count += 1
        
        return retrieved_memory, attention_scores

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
    init_strategy='random'
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

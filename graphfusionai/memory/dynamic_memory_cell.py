import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional

class DynamicMemoryCell(nn.Module):
    """
    Implements a dynamic memory cell that learns and updates stored representations
    using an attention-based mechanism.
    """

    def __init__(self, input_dim: int, memory_dim: int, context_dim: int, lr: float = 0.001, temperature: float = 1.0):
        """
        Initializes the memory cell.

        Args:
            input_dim (int): Dimensionality of input embeddings.
            memory_dim (int): Number of memory slots.
            context_dim (int): Dimensionality of context vectors.
            lr (float): Learning rate for memory updates.
            temperature (float): Scaling factor for attention softmax.
        """
        super(DynamicMemoryCell, self).__init__()
        self.memory_dim = memory_dim
        self.temperature = temperature

        self.keys = nn.Parameter(torch.randn(memory_dim, input_dim))  
        self.values = nn.Parameter(torch.randn(memory_dim, context_dim))  

        self.attention = nn.Linear(input_dim, memory_dim)
        self.update_layer = nn.Linear(context_dim, context_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """
        Retrieves relevant memory using an attention mechanism.

        Args:
            input_embedding (torch.Tensor): Input vector to query memory.

        Returns:
            torch.Tensor: Context vector retrieved from memory.
        """
        attention_scores = torch.softmax(self.attention(input_embedding) / self.temperature, dim=-1)  
        retrieved_memory = torch.matmul(attention_scores, self.values)  
        return retrieved_memory

    def update_memory(self, input_embedding: torch.Tensor, context: torch.Tensor) -> None:
        """
        Updates memory dynamically based on new input.

        Args:
            input_embedding (torch.Tensor): New input representation.
            context (torch.Tensor): Context vector to update memory.
        """
        attention_scores = torch.softmax(self.attention(input_embedding) / self.temperature, dim=-1)
        updated_values = self.update_layer(context)

        loss = torch.nn.functional.mse_loss(self.values, updated_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

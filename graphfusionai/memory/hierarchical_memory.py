import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

@dataclass
class MemoryLevel:
    name: str
    capacity: int
    access_time: float  # milliseconds
    retention_time: float  # hours
    importance_threshold: float

class MemoryHierarchy:
    """
    Implements a hierarchical memory system with multiple levels of storage and processing.
    Inspired by the human memory model with working memory, short-term, and long-term storage.
    """
    
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        
        # Define memory hierarchy levels
        self.levels = {
            "working": MemoryLevel("working", 100, 1.0, 0.1, 0.0),  # 6 minutes retention
            "short_term": MemoryLevel("short_term", 1000, 5.0, 24.0, 0.3),  # 1 day retention
            "long_term": MemoryLevel("long_term", 10000, 20.0, 720.0, 0.7)  # 30 days retention
        }
        
        # Initialize memory stores for each level
        self.stores = {
            level: {
                "data": {},
                "last_access": defaultdict(float),
                "access_count": defaultdict(int),
                "importance": defaultdict(float)
            } for level in self.levels.keys()
        }
        
        # Memory transformers for inter-level transfers
        self.transformers = {
            "compress": MemoryTransformer(vector_dim, "compress"),
            "expand": MemoryTransformer(vector_dim, "expand"),
            "consolidate": MemoryTransformer(vector_dim, "consolidate")
        }
        
    def store(self, key: str, data: Dict[str, Any], importance: float) -> str:
        """Store memory in appropriate hierarchy level based on importance"""
        target_level = self._determine_storage_level(importance)
        
        # Transform data if needed
        if target_level != "working":
            data["embedding"] = self.transformers["compress"](data["embedding"])
            
        # Store in target level
        self.stores[target_level]["data"][key] = data
        self.stores[target_level]["last_access"][key] = datetime.now().timestamp()
        self.stores[target_level]["importance"][key] = importance
        
        return f"{target_level}:{key}"
        
    def retrieve(self, key: str, current_level: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory from hierarchy, handling inter-level transfers"""
        if key not in self.stores[current_level]["data"]:
            # Try to find in other levels
            for level in self.levels.keys():
                if key in self.stores[level]["data"]:
                    data = self.stores[level]["data"][key]
                    # Transfer to target level
                    self._transfer_between_levels(key, level, current_level)
                    return data
            return None
            
        # Update access statistics
        self.stores[current_level]["last_access"][key] = datetime.now().timestamp()
        self.stores[current_level]["access_count"][key] += 1
        
        return self.stores[current_level]["data"][key]
        
    def _determine_storage_level(self, importance: float) -> str:
        """Determine appropriate storage level based on importance"""
        for level_name, level in self.levels.items():
            if importance >= level.importance_threshold:
                return level_name
        return "working"
        
    def _transfer_between_levels(self, key: str, source: str, target: str) -> None:
        """Handle memory transfer between hierarchy levels"""
        data = self.stores[source]["data"][key]
        importance = self.stores[source]["importance"][key]
        
        # Transform data based on transfer direction
        if self.levels[source].capacity > self.levels[target].capacity:
            # Moving to more compressed level
            data["embedding"] = self.transformers["compress"](data["embedding"])
        else:
            # Moving to less compressed level
            data["embedding"] = self.transformers["expand"](data["embedding"])
            
        # Store in target level
        self.stores[target]["data"][key] = data
        self.stores[target]["last_access"][key] = datetime.now().timestamp()
        self.stores[target]["importance"][key] = importance
        
        # Remove from source level
        del self.stores[source]["data"][key]
        
    def consolidate_level(self, level: str) -> None:
        """Consolidate memories within a hierarchy level"""
        data = self.stores[level]["data"]
        if not data:
            return
            
        # Get all embeddings
        keys = list(data.keys())
        embeddings = torch.stack([d["embedding"] for d in data.values()])
        
        # Use transformer to find consolidation groups
        consolidated = self.transformers["consolidate"](embeddings)
        
        # Group similar memories
        groups = defaultdict(list)
        for i, group_id in enumerate(consolidated.argmax(dim=1)):
            groups[group_id.item()].append(keys[i])
            
        # Merge groups
        for group_keys in groups.values():
            if len(group_keys) > 1:
                self._merge_memories(level, group_keys)
                
    def _merge_memories(self, level: str, keys: List[str]) -> None:
        """Merge multiple memories into one"""
        base_key = keys[0]
        base_data = self.stores[level]["data"][base_key]
        
        # Combine metadata and embeddings
        for key in keys[1:]:
            other_data = self.stores[level]["data"][key]
            base_data["metadata"].update(other_data["metadata"])
            base_data["embedding"] = (base_data["embedding"] + other_data["embedding"]) / 2
            
            # Clean up merged memory
            del self.stores[level]["data"][key]
            
    def maintain(self) -> None:
        """Perform maintenance on memory hierarchy"""
        current_time = datetime.now().timestamp()
        
        for level_name, level in self.levels.items():
            store = self.stores[level_name]
            
            # Check for expired memories
            expired_keys = []
            for key, last_access in store["last_access"].items():
                age_hours = (current_time - last_access) / 3600
                if age_hours > level.retention_time:
                    expired_keys.append(key)
                    
            # Remove expired memories
            for key in expired_keys:
                del store["data"][key]
                del store["last_access"][key]
                del store["access_count"][key]
                del store["importance"][key]
                
            # Consolidate level if too full
            if len(store["data"]) > level.capacity * 0.9:
                self.consolidate_level(level_name)
                
class MemoryTransformer(nn.Module):
    """Transformer for memory operations"""
    
    def __init__(self, dim: int, operation: str):
        super().__init__()
        self.operation = operation
        
        if operation == "compress":
            self.model = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, dim)
            )
        elif operation == "expand":
            self.model = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim)
            )
        else:  # consolidate
            self.attention = nn.MultiheadAttention(dim, num_heads=8)
            self.norm = nn.LayerNorm(dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.operation in ["compress", "expand"]:
            return self.model(x)
        else:  # consolidate
            # Self-attention for finding similar memories
            attended, _ = self.attention(x, x, x)
            x = self.norm(x + attended)
            return self.feed_forward(x)

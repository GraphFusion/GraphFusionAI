import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemoryRecord:
    key: str
    vector: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]

class MemoryIndex:
    """Advanced memory indexing system using FAISS for fast similarity search"""
    
    def __init__(self, vector_dim: int, index_type: str = 'IVF', num_clusters: int = 100):
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.num_clusters = num_clusters
        self.records: Dict[str, MemoryRecord] = {}
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize FAISS index based on configuration"""
        if self.index_type == 'IVF':
            # IVF index for large-scale memory
            quantizer = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, 
                                          self.num_clusters, faiss.METRIC_L2)
        elif self.index_type == 'HNSW':
            # HNSW index for fast approximate search
            self.index = faiss.IndexHNSWFlat(self.vector_dim, 32)
        else:
            # Default to flat index for exact search
            self.index = faiss.IndexFlatL2(self.vector_dim)
            
        self.is_trained = False
            
    def add_memory(self, key: str, vector: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """Add a memory vector to the index"""
        vector_np = vector.detach().cpu().numpy().astype('float32')
        
        if not self.is_trained and len(self.records) >= self.num_clusters:
            training_vectors = np.stack([r.vector for r in self.records.values()])
            self.index.train(training_vectors)
            self.is_trained = True
            
        if key in self.records:
            # Update existing record
            old_record = self.records[key]
            self.index.remove_ids(np.array([hash(key) % 2**63]))
            
        record = MemoryRecord(
            key=key,
            vector=vector_np,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.records[key] = record
        
        self.index.add_with_ids(
            vector_np.reshape(1, -1),
            np.array([hash(key) % 2**63])
        )
        
    def search(self, query_vector: torch.Tensor, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar memories"""
        query_np = query_vector.detach().cpu().numpy().astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                for key, record in self.records.items():
                    if hash(key) % 2**63 == idx:
                        results.append((key, float(dist), record.metadata))
                        break
                        
        return results
    
    def remove_memory(self, key: str) -> None:
        """Remove a memory from the index"""
        if key in self.records:
            self.index.remove_ids(np.array([hash(key) % 2**63]))
            del self.records[key]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_memories": len(self.records),
            "index_type": self.index_type,
            "vector_dim": self.vector_dim,
            "is_trained": self.is_trained,
            "memory_usage_bytes": self.index.sa_memory_usage(),
        }

    def optimize(self) -> None:
        """Optimize the index for better performance"""
        if self.index_type == 'IVF':
            # Retrain clusters if needed
            vectors = np.stack([r.vector for r in self.records.values()])
            self.index.train(vectors)
        elif self.index_type == 'HNSW':
            # Optimize HNSW graph
            self.index.hnsw.efConstruction = min(len(self.records), 40)
            self.index.hnsw.efSearch = min(len(self.records), 32)

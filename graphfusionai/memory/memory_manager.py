import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from graphfusionai.memory.memory_index import MemoryIndex
from graphfusionai.memory.dynamic_memory_cell import DynamicMemoryCell
from graphfusionai.memory.embeddings import EmbeddingModel
from graphfusionai.memory.retrieval import MemoryRetrieval
from graphfusionai.memory.hierarchical_memory import MemoryHierarchy
from graphfusionai.memory.advanced_cache import AdvancedCache
from graphfusionai.memory.memory_analytics import MemoryAnalytics
from graphfusionai.memory.attention_system import AttentionSystem
import redis

class MemoryManager:
    """
    Centralized memory controller for GraphFusionAI.
    Handles memory storage, retrieval, and updates with advanced memory management features.
    """

    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 cache_size: int = 1000,
                 consolidation_threshold: float = 0.85,
                 checkpoint_dir: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 vector_dim: int = 768,
                 max_cache_bytes: int = 1024*1024*1024,  # 1GB
                 attention_heads: int = 8):
        """
        Enhanced memory manager with focused attention and hierarchical storage.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Initialize attention system
        self.attention_system = AttentionSystem(
            dim=vector_dim,
            num_heads=attention_heads
        )
        
        # Initialize hierarchical memory
        self.hierarchy = MemoryHierarchy(vector_dim)
        
        # Initialize advanced cache
        self.cache = AdvancedCache(
            max_size_bytes=max_cache_bytes,
            max_items=cache_size,
            policy="adaptive",
            prefetch_threshold=0.8
        )
        
        # Initialize memory index
        self.memory_index = MemoryIndex(
            vector_dim=vector_dim,
            index_type='HNSW'
        )
        
        # Initialize analytics
        self.analytics = MemoryAnalytics(window_size=1000)
        
        # Initialize distributed cache if Redis URL provided
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Local components
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.memory_tags: Dict[str, List[str]] = {}
        self.memory_timestamps: Dict[str, float] = {}
        self.importance_scores: Dict[str, float] = {}
        
        # Configuration
        self.vector_dim = vector_dim
        self.consolidation_threshold = consolidation_threshold
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./memory_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Async components
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = set()
        
        # Start background tasks
        self._schedule_background_task(self._maintain_memory())
        self._schedule_background_task(self.cache.process_prefetch_queue(self._fetch_memory))
        
    def store_memory(self, 
                    data: Union[str, List[str]], 
                    tags: Optional[List[str]] = None,
                    importance: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """
        Stores knowledge as vector embeddings in memory with enhanced metadata.

        Args:
            data: Knowledge to be stored (single text or list of texts)
            tags: Optional list of tags to organize memories
            importance: Importance score (0.0 to 1.0) for memory prioritization
            metadata: Additional context for retrieval

        Returns:
            Unique memory key(s) (single key or list of keys)
        """
        if isinstance(data, str):
            data = [data]
            single_input = True
        else:
            single_input = False

        memory_keys = []
        for text in data:
            vector = self.embedding_model.encode(text)
            memory_key = str(hash(tuple(vector.tolist())))
            
            self.memory_store[memory_key] = {
                "data": text,
                "metadata": metadata or {},
                "embedding": vector
            }
            
            if tags:
                self.memory_tags[memory_key] = tags
                
            self.importance_scores[memory_key] = max(0.0, min(1.0, importance))
            self.memory_timestamps[memory_key] = torch.cuda.Event().record()
            
            self.memory_cell.add(text)
            memory_keys.append(memory_key)

        return memory_keys[0] if single_input else memory_keys

    async def store_memory_async(self, 
                               data: Union[str, List[str]], 
                               tags: Optional[List[str]] = None,
                               importance: float = 1.0,
                               metadata: Optional[Dict[str, Any]] = None,
                               task_id: Optional[int] = None) -> Union[str, List[str]]:
        """Store memory with attention-based importance scoring"""
        if isinstance(data, str):
            data = [data]
            single_input = True
        else:
            single_input = False
            
        memory_keys = []
        start_time = datetime.now().timestamp()
        
        for text in data:
            # Generate embedding
            vector = await self._get_embedding_async(text)
            
            # Apply attention to adjust importance
            if len(self.memory_store) > 0:
                memory_vectors = torch.stack([mem["embedding"] for mem in self.memory_store.values()])
                importance_scores = torch.tensor([self.importance_scores.get(k, 0.0) for k in self.memory_store.keys()])
                
                # Update attention focus with new memory
                adjusted_importance = importance * self.attention_system.update_focus(
                    query_vector=vector,
                    memory_vectors=memory_vectors,
                    importance_scores=importance_scores,
                    timestamps=torch.tensor([start_time])
                ).mean().item()
            else:
                adjusted_importance = importance
            
            memory_key = str(hash(tuple(vector.tolist())))
            
            # Prepare memory data
            memory_data = {
                "data": text,
                "metadata": metadata or {},
                "embedding": vector
            }
            
            # Store with adjusted importance
            hierarchy_key = self.hierarchy.store(memory_key, memory_data, adjusted_importance)
            
            # Store in cache
            size_bytes = sys.getsizeof(text) + vector.element_size() * vector.nelement()
            await self.cache.put(memory_key, memory_data, size_bytes)
            
            # Add to memory index
            self.memory_index.add_memory(memory_key, vector, metadata or {})
            
            # Store in distributed cache if available
            if self.redis_client:
                await self._store_in_cache_async(memory_key, text, vector, metadata)
            
            # Update local tracking
            if tags:
                self.memory_tags[memory_key] = tags
            self.importance_scores[memory_key] = adjusted_importance
            self.memory_timestamps[memory_key] = datetime.now().timestamp()
            
            memory_keys.append(hierarchy_key)
            
            # Update attention context
            self.attention_system.update_context_window(memory_key)
            if task_id is not None:
                self.attention_system.context.active_tasks.append(str(task_id))
            
            # Record analytics
            self.analytics.record_memory_access(
                memory_key=memory_key,
                query_latency_ms=(datetime.now().timestamp() - start_time) * 1000,
                cache_hit=False,
                importance=adjusted_importance
            )
            
        return memory_keys[0] if single_input else memory_keys

    async def retrieve_memory_async(self,
                                  query: str,
                                  top_k: int = 5,
                                  min_similarity: float = 0.0,
                                  tags: Optional[List[str]] = None,
                                  strategy: str = 'hybrid',
                                  task_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enhanced memory retrieval with focused attention"""
        start_time = datetime.now().timestamp()
        query_vector = await self._get_embedding_async(query)
        
        # Update attention focus
        if len(self.memory_store) > 0:
            memory_vectors = torch.stack([mem["embedding"] for mem in self.memory_store.values()])
            memory_keys = list(self.memory_store.keys())
            importance_scores = torch.tensor([self.importance_scores.get(k, 0.0) for k in memory_keys])
            timestamps = torch.tensor([self.memory_timestamps.get(k, 0.0) for k in memory_keys])
            
            # Update attention focus
            focus_vector = self.attention_system.update_focus(
                query_vector=query_vector,
                memory_vectors=memory_vectors,
                importance_scores=importance_scores,
                timestamps=timestamps
            )
            
            # Apply task-specific attention if task_id provided
            if task_id is not None:
                memory_vectors = self.attention_system.apply_task_attention(
                    memory_vectors=memory_vectors,
                    task_ids=[task_id]
                )
            
            # Get focused memories
            focused_memories, attention_weights = self.attention_system.get_focused_memories(
                memory_vectors=memory_vectors,
                query_vector=query_vector,
                top_k=top_k
            )
            
            # Convert focused memories to results
            results = []
            for i, (memory_vec, weight) in enumerate(zip(focused_memories, attention_weights)):
                # Find matching memory
                for key, memory in self.memory_store.items():
                    if torch.allclose(memory["embedding"], memory_vec):
                        results.append({
                            "key": key,
                            "data": memory["data"],
                            "similarity": weight.item(),
                            "metadata": memory["metadata"]
                        })
                        # Update context window
                        self.attention_system.update_context_window(key)
                        break
            
            # Record analytics
            query_time = (datetime.now().timestamp() - start_time) * 1000
            self.analytics.record_memory_access(
                memory_key="query",
                query_latency_ms=query_time,
                cache_hit=len(results) > 0,
                importance=1.0
            )
            
            # Analyze attention
            attention_stats = self.attention_system.analyze_attention()
            self.analytics.record_attention_metrics(attention_stats)
            
            return results[:top_k]
            
        return []

    async def _get_embedding_async(self, text: str) -> torch.Tensor:
        """Get embedding asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            lambda: self.embedding_model.encode(text)
        )
        
    async def _store_in_cache_async(self, key: str, text: str, vector: torch.Tensor, metadata: Dict[str, Any]):
        """Store memory in distributed cache"""
        if self.redis_client:
            cache_data = {
                "text": text,
                "vector": vector.tolist(),
                "metadata": metadata
            }
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.redis_client.set(f"memory:{key}", json.dumps(cache_data))
            )
            
    async def _get_memory_data_async(self, key: str) -> Optional[Dict[str, Any]]:
        """Get memory data from cache or local storage"""
        # Try distributed cache first
        if self.redis_client:
            cache_data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.redis_client.get(f"memory:{key}")
            )
            if cache_data:
                return json.loads(cache_data)
                
        # Fall back to local storage
        return self.memory_store.get(key)
        
    async def _exact_search(self, query_vector: torch.Tensor, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform exact similarity search"""
        results = []
        for key, memory in self.memory_store.items():
            similarity = torch.cosine_similarity(query_vector, memory["embedding"], dim=0).item()
            results.append((key, similarity, memory["metadata"]))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
        
    def _merge_results(self, exact_results: List[Tuple], approx_results: List[Tuple]) -> List[Tuple]:
        """Merge and deduplicate search results"""
        seen_keys = set()
        merged = []
        
        for results in [exact_results, approx_results]:
            for key, similarity, metadata in results:
                if key not in seen_keys:
                    merged.append((key, similarity, metadata))
                    seen_keys.add(key)
                    
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged
        
    def _schedule_background_task(self, coro):
        """Schedule a background task"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
    async def _maintain_memory(self):
        """Background task for memory maintenance"""
        while True:
            # Maintain hierarchy
            self.hierarchy.maintain()
            
            # Optimize memory index
            self.memory_index.optimize()
            
            # Generate analytics report
            report = self.analytics.generate_analytics_report()
            
            # Adjust cache parameters based on analytics
            if report["current_metrics"]["cache_hit_rate"] < 0.5:
                # Increase cache size or adjust prefetch threshold
                self.cache.prefetch_threshold *= 0.95
            
            await asyncio.sleep(60)  # Run maintenance every minute
            
    async def _fetch_memory(self, key: str) -> Optional[Tuple[Dict[str, Any], int]]:
        """Fetch memory for cache prefetching"""
        # Try hierarchy first
        for level in ["working", "short_term", "long_term"]:
            memory_data = self.hierarchy.retrieve(key, level)
            if memory_data:
                size_bytes = (sys.getsizeof(memory_data["data"]) + 
                            memory_data["embedding"].element_size() * 
                            memory_data["embedding"].nelement())
                return memory_data, size_bytes
        return None
    
    def retrieve_memory(self, 
                       query: str,
                       top_k: int = 5,
                       min_similarity: float = 0.0,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant memories based on the query with filtering options.

        Args:
            query: The input query for retrieval
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold for results
            tags: Optional list of tags to filter memories

        Returns:
            List of relevant memory entries with similarity scores
        """
        results = self.memory_cell.query(query, top_k=top_k)
        
        results = [r for r in results if r["similarity"] >= min_similarity]
        
        if tags:
            filtered_results = []
            for result in results:
                memory_key = self._get_key_by_text(result["text"])
                if memory_key and any(tag in self.memory_tags.get(memory_key, []) for tag in tags):
                    filtered_results.append(result)
            results = filtered_results

        for result in results:
            memory_key = self._get_key_by_text(result["text"])
            if memory_key:
                result["metadata"] = self.memory_store[memory_key].get("metadata", {})
                result["importance"] = self.importance_scores.get(memory_key, 1.0)
                
        return results

    def update_memory(self, 
                     memory_key: str, 
                     new_data: Optional[str] = None,
                     new_tags: Optional[List[str]] = None,
                     new_importance: Optional[float] = None,
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Updates an existing memory entry with flexible field updates.

        Args:
            memory_key: The key of the memory to update
            new_data: Optional updated information
            new_tags: Optional updated tags
            new_importance: Optional updated importance score
            new_metadata: Optional updated metadata

        Returns:
            Whether the update was successful
        """
        if memory_key not in self.memory_store:
            return False
            
        if new_data:
            old_data = self.memory_store[memory_key]["data"]
            self.memory_cell.remove(old_data)  
            self.memory_store[memory_key]["data"] = new_data
            self.memory_store[memory_key]["embedding"] = self.embedding_model.encode(new_data)
            self.memory_cell.add(new_data)  
            
        if new_tags is not None:
            self.memory_tags[memory_key] = new_tags
            
        if new_importance is not None:
            self.importance_scores[memory_key] = max(0.0, min(1.0, new_importance))
            
        if new_metadata is not None:
            self.memory_store[memory_key]["metadata"].update(new_metadata)
            
        return True

    def _get_key_by_text(self, text: str) -> Optional[str]:
        """Helper method to find memory key by stored text."""
        for key, value in self.memory_store.items():
            if value["data"] == text:
                return key
        return None

    def consolidate_memories(self) -> None:
        """Consolidate similar memories to optimize storage"""
        all_vectors = torch.stack([mem["embedding"] for mem in self.memory_store.values()])
        similarities = torch.matmul(all_vectors, all_vectors.T)
        
        consolidated = set()
        for i in range(len(similarities)):
            if i in consolidated:
                continue
                
            similar_indices = torch.where(similarities[i] > self.consolidation_threshold)[0]
            if len(similar_indices) > 1:
                self._merge_memories(similar_indices.tolist())
                consolidated.update(similar_indices.tolist())

    async def consolidate_memories_async(self) -> None:
        """Consolidate similar memories to optimize storage"""
        all_vectors = torch.stack([mem["embedding"] for mem in self.memory_store.values()])
        similarities = torch.matmul(all_vectors, all_vectors.T)
        
        consolidated = set()
        for i in range(len(similarities)):
            if i in consolidated:
                continue
                
            similar_indices = torch.where(similarities[i] > self.consolidation_threshold)[0]
            if len(similar_indices) > 1:
                await self._merge_memories_async(similar_indices.tolist())
                consolidated.update(similar_indices.tolist())

    def _merge_memories(self, indices: List[int]) -> None:
        """Merge similar memories into a single, more comprehensive memory"""
        keys = list(self.memory_store.keys())
        base_key = keys[indices[0]]
        
        # Merge metadata and tags
        for idx in indices[1:]:
            key = keys[idx]
            self.memory_store[base_key]["metadata"].update(self.memory_store[key]["metadata"])
            if key in self.memory_tags:
                self.memory_tags[base_key] = list(set(self.memory_tags[base_key] + self.memory_tags[key]))
            
            # Clean up merged memory
            del self.memory_store[key]
            self.memory_tags.pop(key, None)
            self.memory_timestamps.pop(key, None)
            self.importance_scores.pop(key, None)

    async def _merge_memories_async(self, indices: List[int]) -> None:
        """Merge similar memories into a single, more comprehensive memory"""
        keys = list(self.memory_store.keys())
        base_key = keys[indices[0]]
        
        # Merge metadata and tags
        for idx in indices[1:]:
            key = keys[idx]
            self.memory_store[base_key]["metadata"].update(self.memory_store[key]["metadata"])
            if key in self.memory_tags:
                self.memory_tags[base_key] = list(set(self.memory_tags[base_key] + self.memory_tags[key]))
            
            # Clean up merged memory
            del self.memory_store[key]
            self.memory_tags.pop(key, None)
            self.memory_timestamps.pop(key, None)
            self.importance_scores.pop(key, None)

    def checkpoint_memory(self) -> str:
        """Save current memory state to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"memory_checkpoint_{timestamp}.pkl"
        
        state = {
            "memory_store": self.memory_store,
            "memory_tags": self.memory_tags,
            "memory_timestamps": self.memory_timestamps,
            "importance_scores": self.importance_scores
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
            
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load memory state from checkpoint"""
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
            
        self.memory_store = state["memory_store"]
        self.memory_tags = state["memory_tags"]
        self.memory_timestamps = state["memory_timestamps"]
        self.importance_scores = state["importance_scores"]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "total_memories": len(self.memory_store),
            "tagged_memories": len(self.memory_tags),
            "avg_importance": sum(self.importance_scores.values()) / len(self.importance_scores) if self.importance_scores else 0,
            "memory_size_bytes": sys.getsizeof(pickle.dumps(self.memory_store)),
            "cache_usage": len(self.cache) / self.cache_size
        }

    def prune_memories(self, age_threshold: float = 7*24*60*60, importance_threshold: float = 0.3) -> int:
        """Remove old and low-importance memories"""
        current_time = torch.cuda.Event().record()
        keys_to_remove = []
        
        for key, timestamp in self.memory_timestamps.items():
            age = current_time - timestamp
            importance = self.importance_scores.get(key, 0)
            
            if age > age_threshold and importance < importance_threshold:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.memory_store[key]
            self.memory_tags.pop(key, None)
            self.memory_timestamps.pop(key, None)
            self.importance_scores.pop(key, None)
            
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clears all stored memories and related data."""
        self.memory_store.clear()
        self.memory_tags.clear()
        self.memory_timestamps.clear()
        self.importance_scores.clear()
        self.memory_cell.clear()

    def search_memories(self, query: str, top_k: int = 3, min_similarity: float = 0.0, 
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search memories using the query string.
        
        Args:
            query: The search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching memories with their similarity scores
        """
        query_vector = self.embedding_model.encode(query)
        return self.retrieval.search(
            self.memory_store,
            query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata
        )

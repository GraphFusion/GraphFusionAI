import torch
from typing import Dict, List, Any, Optional, Tuple, Generic, TypeVar
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import heapq

T = TypeVar('T')

@dataclass
class CacheItem(Generic[T]):
    key: str
    value: T
    score: float
    last_access: float
    access_count: int
    size_bytes: int

class AdvancedCache:
    """
    Advanced caching system with multiple eviction policies and predictive prefetching.
    Features:
    - Multiple eviction policies (LRU, LFU, GDSF)
    - Score-based retention
    - Predictive prefetching
    - Memory-aware sizing
    - Thread-safe operations
    """
    
    def __init__(self, 
                 max_size_bytes: int,
                 max_items: int = 10000,
                 policy: str = "adaptive",
                 prefetch_threshold: float = 0.8):
        self.max_size_bytes = max_size_bytes
        self.max_items = max_items
        self.policy = policy
        self.prefetch_threshold = prefetch_threshold
        
        # Main cache storage
        self.cache: Dict[str, CacheItem] = {}
        self.size_bytes = 0
        
        # Access patterns for prediction
        self.access_patterns: Dict[str, List[str]] = {}
        self.pattern_scores: Dict[Tuple[str, str], float] = {}
        
        # Eviction policy scores
        self.policy_scores = {
            "lru": 1.0,
            "lfu": 1.0,
            "gdsf": 1.0
        }
        
        # Threading
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Prefetch queue
        self.prefetch_queue = []
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with predictive prefetching"""
        with self.lock:
            item = self.cache.get(key)
            if item:
                # Update access statistics
                current_time = datetime.now().timestamp()
                item.last_access = current_time
                item.access_count += 1
                
                # Update access patterns
                self._update_access_pattern(key)
                
                # Schedule prefetching
                await self._schedule_prefetch(key)
                
                return item.value
            return None
            
    async def put(self, key: str, value: Any, size_bytes: int) -> None:
        """Put item in cache with smart eviction"""
        with self.lock:
            # Check if we need to evict
            while (self.size_bytes + size_bytes > self.max_size_bytes or 
                   len(self.cache) >= self.max_items):
                if not self._evict_one():
                    # Cannot evict more items
                    return
                    
            # Create new cache item
            item = CacheItem(
                key=key,
                value=value,
                score=0.0,
                last_access=datetime.now().timestamp(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            self.cache[key] = item
            self.size_bytes += size_bytes
            
            # Update item score
            self._update_item_score(item)
            
    def _evict_one(self) -> bool:
        """Evict one item based on current policy"""
        if not self.cache:
            return False
            
        if self.policy == "adaptive":
            # Use policy with highest score
            policy = max(self.policy_scores.items(), key=lambda x: x[1])[0]
        else:
            policy = self.policy
            
        # Get item to evict
        to_evict = None
        if policy == "lru":
            to_evict = min(self.cache.values(), key=lambda x: x.last_access)
        elif policy == "lfu":
            to_evict = min(self.cache.values(), key=lambda x: x.access_count)
        else:  # gdsf
            to_evict = min(self.cache.values(), key=lambda x: x.score)
            
        # Remove item
        self.size_bytes -= to_evict.size_bytes
        del self.cache[to_evict.key]
        
        return True
        
    def _update_item_score(self, item: CacheItem) -> None:
        """Update item score using GDSF policy"""
        current_time = datetime.now().timestamp()
        age = current_time - item.last_access
        
        # Calculate scores for different policies
        lru_score = 1.0 / (age + 1)
        lfu_score = item.access_count
        gdsf_score = (item.access_count * item.size_bytes) / (age + 1)
        
        # Update policy scores based on hit/miss
        if item.access_count > 1:
            self.policy_scores["lru"] *= 0.95
            self.policy_scores["lfu"] *= 1.05
            self.policy_scores["gdsf"] *= 1.02
            
        # Set item score based on current policy
        if self.policy == "lru":
            item.score = lru_score
        elif self.policy == "lfu":
            item.score = lfu_score
        else:
            item.score = gdsf_score
            
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern predictions"""
        for prev_key in list(self.cache.keys())[-5:]:  # Look at last 5 accesses
            if prev_key != key:
                pattern = (prev_key, key)
                self.pattern_scores[pattern] = self.pattern_scores.get(pattern, 0) * 0.95 + 0.05
                
    async def _schedule_prefetch(self, key: str) -> None:
        """Schedule predictive prefetching of likely next items"""
        # Find items likely to be accessed next
        candidates = []
        for pattern, score in self.pattern_scores.items():
            if pattern[0] == key and score > self.prefetch_threshold:
                candidates.append((score, pattern[1]))
                
        # Sort by probability
        candidates.sort(reverse=True)
        
        # Schedule top candidates for prefetching
        for _, next_key in candidates[:3]:  # Prefetch top 3 candidates
            if next_key not in self.cache:
                heapq.heappush(self.prefetch_queue, (-self.pattern_scores[(key, next_key)], next_key))
                
    async def process_prefetch_queue(self, fetch_func) -> None:
        """Process prefetch queue in background"""
        while True:
            if self.prefetch_queue:
                _, key = heapq.heappop(self.prefetch_queue)
                if key not in self.cache:
                    # Fetch item using provided function
                    value = await fetch_func(key)
                    if value is not None:
                        await self.put(key, value[0], value[1])  # value is (data, size)
            await asyncio.sleep(0.1)  # Avoid busy waiting
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            current_time = datetime.now().timestamp()
            return {
                "size_bytes": self.size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "item_count": len(self.cache),
                "max_items": self.max_items,
                "policy_scores": self.policy_scores.copy(),
                "avg_access_count": np.mean([item.access_count for item in self.cache.values()]),
                "avg_age": np.mean([current_time - item.last_access for item in self.cache.values()]),
                "pattern_count": len(self.pattern_scores),
                "prefetch_queue_size": len(self.prefetch_queue)
            }

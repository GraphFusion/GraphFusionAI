import numpy as np
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from collections import defaultdict

@dataclass
class MemoryMetrics:
    total_memories: int
    active_memories: int
    memory_usage_bytes: int
    avg_importance: float
    cache_hit_rate: float
    query_latency_ms: float
    compression_ratio: float

class MemoryAnalytics:
    """Advanced memory analytics and visualization system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: List[Dict[str, Any]] = []
        self.access_patterns = defaultdict(list)
        self.query_latencies = []
        self.importance_distribution = []
        
    def record_memory_access(self, memory_key: str, query_latency_ms: float,
                           cache_hit: bool, importance: float):
        """Record a memory access event"""
        timestamp = datetime.now()
        
        self.metrics_history.append({
            "timestamp": timestamp,
            "memory_key": memory_key,
            "query_latency_ms": query_latency_ms,
            "cache_hit": cache_hit,
            "importance": importance
        })
        
        # Maintain window size
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
            
        # Record access patterns
        self.access_patterns[memory_key].append(timestamp)
        self.query_latencies.append(query_latency_ms)
        self.importance_distribution.append(importance)
        
    def get_current_metrics(self) -> MemoryMetrics:
        """Calculate current memory metrics"""
        if not self.metrics_history:
            return MemoryMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
            
        recent_records = self.metrics_history[-min(100, len(self.metrics_history)):]
        
        return MemoryMetrics(
            total_memories=len(self.access_patterns),
            active_memories=len([k for k, v in self.access_patterns.items()
                               if (datetime.now() - v[-1]) < timedelta(hours=24)]),
            memory_usage_bytes=sum(sys.getsizeof(str(m)) for m in recent_records),
            avg_importance=np.mean(self.importance_distribution[-100:]),
            cache_hit_rate=np.mean([r["cache_hit"] for r in recent_records]),
            query_latency_ms=np.mean(self.query_latencies[-100:]),
            compression_ratio=0.0  # Set by memory manager
        )
        
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        metrics = self.get_current_metrics()
        
        # Calculate temporal patterns
        hourly_access = pd.DataFrame(self.metrics_history)
        if not hourly_access.empty:
            hourly_access["hour"] = hourly_access["timestamp"].dt.hour
            hourly_pattern = hourly_access.groupby("hour").size()
        else:
            hourly_pattern = pd.Series()
            
        # Calculate memory importance distribution
        importance_hist = np.histogram(self.importance_distribution, bins=10)
        
        return {
            "current_metrics": metrics.__dict__,
            "temporal_patterns": {
                "hourly_access": hourly_pattern.to_dict(),
                "peak_hours": hourly_pattern.nlargest(3).index.tolist()
            },
            "performance_metrics": {
                "avg_latency": np.mean(self.query_latencies),
                "p95_latency": np.percentile(self.query_latencies, 95),
                "latency_trend": self._calculate_trend(self.query_latencies)
            },
            "importance_distribution": {
                "histogram": {
                    "counts": importance_hist[0].tolist(),
                    "bins": importance_hist[1].tolist()
                },
                "statistics": {
                    "mean": np.mean(self.importance_distribution),
                    "median": np.median(self.importance_distribution),
                    "std": np.std(self.importance_distribution)
                }
            },
            "memory_health": self._assess_memory_health(metrics)
        }
    
    def visualize_metrics(self) -> Dict[str, Any]:
        """Generate interactive visualizations of memory metrics"""
        df = pd.DataFrame(self.metrics_history)
        
        if df.empty:
            return {}
            
        # Latency over time
        latency_fig = px.line(df, x="timestamp", y="query_latency_ms",
                             title="Memory Query Latency Over Time")
        
        # Importance distribution
        importance_fig = px.histogram(df, x="importance",
                                    title="Memory Importance Distribution")
        
        # Cache hit rate over time
        cache_fig = px.line(df.rolling(window=50).mean(),
                           x="timestamp", y="cache_hit",
                           title="Cache Hit Rate (50-point moving average)")
        
        return {
            "latency_plot": latency_fig.to_json(),
            "importance_plot": importance_fig.to_json(),
            "cache_plot": cache_fig.to_json()
        }
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calculate trend direction from recent values"""
        if len(values) < window * 2:
            return "insufficient_data"
            
        recent = np.mean(values[-window:])
        previous = np.mean(values[-2*window:-window])
        
        if recent < previous * 0.95:
            return "improving"
        elif recent > previous * 1.05:
            return "degrading"
        else:
            return "stable"
    
    def _assess_memory_health(self, metrics: MemoryMetrics) -> Dict[str, Any]:
        """Assess overall memory system health"""
        health_metrics = {
            "status": "healthy",
            "warnings": [],
            "recommendations": []
        }
        
        # Check cache hit rate
        if metrics.cache_hit_rate < 0.8:
            health_metrics["warnings"].append("Low cache hit rate")
            health_metrics["recommendations"].append(
                "Consider increasing cache size or adjusting caching strategy"
            )
            
        # Check query latency
        if metrics.query_latency_ms > 100:
            health_metrics["warnings"].append("High query latency")
            health_metrics["recommendations"].append(
                "Consider optimizing index or reducing memory size"
            )
            
        # Check memory usage
        if metrics.memory_usage_bytes > 1e9:  # 1GB
            health_metrics["warnings"].append("High memory usage")
            health_metrics["recommendations"].append(
                "Consider increasing compression ratio or pruning old memories"
            )
            
        # Update overall status
        if len(health_metrics["warnings"]) > 2:
            health_metrics["status"] = "critical"
        elif len(health_metrics["warnings"]) > 0:
            health_metrics["status"] = "warning"
            
        return health_metrics

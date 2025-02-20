from typing import Dict, Type, Optional, List, Any
from collections import defaultdict
import asyncio
import time
from .base import BaseTool

class ToolRegistry:
    """An enhanced registry to manage and optimize tool usage."""
    
    _instance: Optional['ToolRegistry'] = None
    _tools: Dict[str, Type[BaseTool]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_registry()
        return cls._instance
    
    def _init_registry(self):
        """Initialize registry components."""
        self._tools = {}
        self._categories = defaultdict(list)
        self._usage_stats = defaultdict(lambda: {
            'total_calls': 0,
            'success_calls': 0,
            'total_duration': 0.0,
            'last_used': None
        })
        self._tool_dependencies = defaultdict(set)
        self._tool_instances = {}
        
    def register(self, tool_class: Type[BaseTool], category: str = "general",
                dependencies: List[str] = None) -> None:
        """
        Register a tool class with enhanced metadata.
        
        Args:
            tool_class: The tool class to register
            category: Tool category for organization
            dependencies: List of required tool names
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError("Tool class must inherit from BaseTool")
            
        tool_name = tool_class.__name__.lower()
        
        # Register tool
        self._tools[tool_name] = tool_class
        self._categories[category].append(tool_name)
        
        # Track dependencies
        if dependencies:
            self._tool_dependencies[tool_name].update(dependencies)
            
            # Verify dependencies
            missing = [dep for dep in dependencies if dep not in self._tools]
            if missing:
                raise ValueError(f"Missing dependencies for {tool_name}: {missing}")
    
    async def get(self, name: str, init_params: Dict[str, Any] = None) -> BaseTool:
        """
        Get or create a tool instance with caching.
        
        Args:
            name: Tool name
            init_params: Optional initialization parameters
            
        Returns:
            Tool instance
        """
        name = name.lower()
        
        # Check if instance exists
        if name in self._tool_instances:
            return self._tool_instances[name]
            
        # Get tool class
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
            
        # Create new instance
        tool_class = self._tools[name]
        instance = tool_class(**(init_params or {}))
        
        # Initialize dependencies
        for dep_name in self._tool_dependencies[name]:
            dep_instance = await self.get(dep_name)
            setattr(instance, f"_{dep_name}", dep_instance)
        
        # Cache instance
        self._tool_instances[name] = instance
        return instance
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """
        List tools organized by category.
        
        Returns:
            Dictionary mapping categories to tool lists
        """
        return dict(self._categories)
    
    def get_tool_stats(self, name: str) -> Dict[str, Any]:
        """
        Get usage statistics for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool statistics
        """
        name = name.lower()
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
            
        stats = self._usage_stats[name]
        if stats['total_calls'] > 0:
            success_rate = stats['success_calls'] / stats['total_calls']
            avg_duration = stats['total_duration'] / stats['total_calls']
        else:
            success_rate = 0.0
            avg_duration = 0.0
            
        return {
            'total_calls': stats['total_calls'],
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'last_used': stats['last_used']
        }
    
    def update_stats(self, name: str, duration: float, success: bool) -> None:
        """
        Update usage statistics for a tool.
        
        Args:
            name: Tool name
            duration: Execution duration
            success: Whether execution was successful
        """
        stats = self._usage_stats[name]
        stats['total_calls'] += 1
        if success:
            stats['success_calls'] += 1
        stats['total_duration'] += duration
        stats['last_used'] = time.time()
    
    def get_recommended_tools(self, task_description: str) -> List[str]:
        """
        Get tool recommendations based on task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of recommended tool names
        """
        # This is a placeholder for more sophisticated recommendation logic
        recommendations = []
        for name, tool_class in self._tools.items():
            if any(word.lower() in task_description.lower() 
                  for word in tool_class.description.split()):
                recommendations.append(name)
        return recommendations
    
    def optimize_tool_usage(self) -> Dict[str, Any]:
        """
        Analyze and optimize tool usage patterns.
        
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'caching': {},
            'timeout': {},
            'retries': {}
        }
        
        for name, stats in self._usage_stats.items():
            if stats['total_calls'] > 0:
                # Recommend caching for frequently used tools
                if stats['total_calls'] > 100 and stats['success_rate'] > 0.9:
                    recommendations['caching'][name] = True
                
                # Recommend timeout adjustments
                if stats['avg_duration'] > 5.0:
                    recommendations['timeout'][name] = stats['avg_duration'] * 1.5
                
                # Recommend retry settings
                if 0.3 < stats['success_rate'] < 0.8:
                    recommendations['retries'][name] = min(5, int(1 / stats['success_rate']))
        
        return recommendations
    
    def clear(self) -> None:
        """Clear all registry data."""
        self._init_registry()

# Create global registry instance
registry = ToolRegistry()
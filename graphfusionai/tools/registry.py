"""
Tool registry with enhanced features and async support.
"""
from typing import Dict, Type, Optional, List, Any, Set
from collections import defaultdict
import asyncio
import time
import logging
from .base import BaseTool, ToolError, ToolConfig

logger = logging.getLogger(__name__)

class RegistryError(ToolError):
    """Base exception for registry-related errors."""
    pass

class DependencyError(RegistryError):
    """Raised when tool dependencies cannot be satisfied."""
    pass

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
        self._tool_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._tool_instances: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}
        self.logger = logging.getLogger("tool.registry")
        
    def register(self, tool_class: Type[BaseTool], category: str = "general",
                dependencies: List[str] = None, config: Optional[ToolConfig] = None) -> None:
        """
        Register a tool class with enhanced metadata.
        
        Args:
            tool_class: The tool class to register
            category: Tool category for organization
            dependencies: List of required tool names
            config: Optional tool configuration
            
        Raises:
            TypeError: When tool_class is not a BaseTool subclass
            ValueError: When tool name conflicts or dependencies are invalid
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError("Tool class must inherit from BaseTool")
            
        tool_name = tool_class.__name__.lower()
        
        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name} is already registered")
        
        # Register tool
        self._tools[tool_name] = tool_class
        self._categories[category].append(tool_name)
        
        # Store configuration
        if config:
            self._tool_configs[tool_name] = config
        
        # Track dependencies
        if dependencies:
            missing = [d for d in dependencies if d not in self._tools]
            if missing:
                raise ValueError(f"Missing dependencies: {', '.join(missing)}")
            self._tool_dependencies[tool_name].update(dependencies)
            
        self.logger.info(f"Registered tool {tool_name} in category {category}")
        
    async def get_tool(self, tool_name: str) -> BaseTool:
        """
        Get or create a tool instance with dependency resolution.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: When tool is not found
            DependencyError: When dependencies cannot be satisfied
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool {tool_name} not found")
            
        # Return existing instance if available
        if tool_name in self._tool_instances:
            return self._tool_instances[tool_name]
            
        # Create new instance with dependencies
        tool_class = self._tools[tool_name]
        config = self._tool_configs.get(tool_name)
        
        # Resolve dependencies
        dependencies = {}
        for dep_name in self._tool_dependencies[tool_name]:
            try:
                dependencies[dep_name] = await self.get_tool(dep_name)
            except Exception as e:
                raise DependencyError(f"Failed to resolve dependency {dep_name}: {str(e)}")
                
        # Create instance
        instance = tool_class(config=config, dependencies=dependencies)
        self._tool_instances[tool_name] = instance
        return instance
        
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool and clean up its resources.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self._tool_instances:
            instance = self._tool_instances[tool_name]
            asyncio.create_task(instance.__aexit__(None, None, None))
            del self._tool_instances[tool_name]
            
        if tool_name in self._tools:
            del self._tools[tool_name]
            
        if tool_name in self._tool_configs:
            del self._tool_configs[tool_name]
            
        for category in self._categories.values():
            if tool_name in category:
                category.remove(tool_name)
                
        if tool_name in self._tool_dependencies:
            del self._tool_dependencies[tool_name]
            
        self.logger.info(f"Unregistered tool {tool_name}")
        
    def get_categories(self) -> Dict[str, List[str]]:
        """Get all tool categories and their tools."""
        return dict(self._categories)
        
    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed tool metadata.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool metadata including usage stats
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool {tool_name} not found")
            
        metadata = {}
        if tool_name in self._tool_instances:
            metadata = self._tool_instances[tool_name].get_metadata()
            
        metadata.update({
            "category": next(
                (cat for cat, tools in self._categories.items() if tool_name in tools),
                "unknown"
            ),
            "dependencies": list(self._tool_dependencies[tool_name]),
            "usage_stats": self._usage_stats[tool_name]
        })
        
        return metadata
        
    def clear_caches(self) -> None:
        """Clear caches of all tool instances."""
        for instance in self._tool_instances.values():
            instance.clear_cache()
            
    async def shutdown(self) -> None:
        """Clean up all tool instances."""
        for instance in self._tool_instances.values():
            await instance.__aexit__(None, None, None)
        self._tool_instances.clear()

# Create global registry instance
registry = ToolRegistry()
from typing import Dict, Type, Optional
from .base import BaseTool

class ToolRegistry:
    """A registry to manage and access available tools."""
    
    _instance: Optional['ToolRegistry'] = None
    _tools: Dict[str, Type[BaseTool]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class.
        
        Args:
            tool_class: The tool class to register, must inherit from BaseTool
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError("Tool class must inherit from BaseTool")
            
        tool_name = tool_class.__name__.lower()
        self._tools[tool_name] = tool_class

    def get(self, name: str) -> Type[BaseTool]:
        """
        Get a registered tool class by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The requested tool class
            
        Raises:
            KeyError: If tool is not found
        """
        name = name.lower()
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list(self) -> Dict[str, str]:
        """
        List all registered tools and their descriptions.
        
        Returns:
            Dictionary mapping tool names to their descriptions
        """
        return {
            name: tool_class.description 
            for name, tool_class in self._tools.items()
        }

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
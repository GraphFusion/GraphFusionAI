from typing import Any, Dict, Optional

class BaseTool:
    """
    Abstract base class for tools that can be used by agents.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
        """
        self.name = name
        self.description = description

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's functionality.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of executing the tool

        Raises:
            NotImplementedError: If subclass doesn't implement execute
        """
        raise NotImplementedError("Subclasses must implement the 'execute' method.")

    def get_metadata(self) -> Dict[str, str]:
        """
        Get metadata about the tool.

        Returns:
            Dictionary containing tool metadata like name and description
        """
        return {
            "name": self.name,
            "description": self.description
        }
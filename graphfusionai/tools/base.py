
from typing import Any

class BaseTool:
    """
    Abstract base class for tools.
    """
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's functionality.
        """
        raise NotImplementedError("Subclasses must implement the 'execute' method.")
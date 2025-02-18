import requests
from graphfusionai.tools.base import BaseTool
from graphfusionai.tools.registry import ToolRegistry

class WebTool(BaseTool):
    """
    Tool for web-based actions, such as making HTTP requests.
    """
    def __init__(self):
        super().__init__(
            name="web",
            description="Makes HTTP requests to web URLs and returns the response content"
        )

    def execute(self, url: str, method: str = "GET", headers: dict = None, data: dict = None) -> str:
        """
        Execute a web request.

        Args:
            url: The URL to send the request to
            method: HTTP method to use (GET, POST, etc)
            headers: Optional headers to include
            data: Optional data payload for POST/PUT requests

        Returns:
            The response text content
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        headers = headers or {}
        data = data or {}

        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=data if method.upper() in ["POST", "PUT", "PATCH"] else None
        )
        response.raise_for_status()
        return response.text

# Register the tool
registry = ToolRegistry()
registry.register(WebTool)

"""
GraphFusionAI tools package.
"""
from .base import BaseTool, ToolConfig, ToolError
from .registry import registry
from .standard.web_search import WebSearchTool, WebSearchConfig

# Register standard tools
registry.register(
    WebSearchTool,
    category="web",
    config=WebSearchConfig(
        name="web_search",
        description="Search the web using Tavily API",
        api_key="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"
    )
)

__all__ = [
    "BaseTool",
    "ToolConfig",
    "ToolError",
    "registry",
    "WebSearchTool",
    "WebSearchConfig"
]
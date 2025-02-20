"""
Web search tool using Tavily API.
"""
from typing import Any, Dict, List, Optional
from pydantic import Field
import asyncio
from tavily import TavilyClient
from ..base import BaseTool, ToolConfig, ToolError

class WebSearchConfig(ToolConfig):
    """Configuration for web search tool."""
    api_key: str = Field(..., description="Tavily API key")
    search_depth: str = Field(default="basic", description="Search depth (basic or advanced)")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    include_images: bool = Field(default=False, description="Whether to include images in results")
    include_answer: bool = Field(default=True, description="Whether to include AI-generated answer")
    
class WebSearchError(ToolError):
    """Raised when web search fails."""
    pass

class WebSearchTool(BaseTool):
    """Tool for performing web searches using Tavily API."""
    
    def __init__(self, config: Optional[WebSearchConfig] = None, **kwargs):
        """Initialize web search tool."""
        if config is None:
            config = WebSearchConfig(
                name="web_search",
                description="Search the web using Tavily API",
                api_key=kwargs.pop("api_key", None),
                **kwargs
            )
        super().__init__(config=config)
        
        if not config.api_key:
            raise WebSearchError("Tavily API key is required")
            
        self.client = TavilyClient(api_key=config.api_key)
        
    async def _execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute web search.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Search results including URLs, snippets, and optionally an AI-generated answer
            
        Raises:
            WebSearchError: When search fails
        """
        try:
            # Get search parameters from config and kwargs
            params = {
                "query": query,
                "search_depth": kwargs.get("search_depth", self.config.search_depth),
                "max_results": kwargs.get("max_results", self.config.max_results),
                "include_images": kwargs.get("include_images", self.config.include_images),
                "include_answer": kwargs.get("include_answer", self.config.include_answer)
            }
            
            # Execute search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.client.search(**params)
            )
            
            return {
                "query": query,
                "results": results.get("results", []),
                "answer": results.get("answer"),
                "context": results.get("context")
            }
            
        except Exception as e:
            raise WebSearchError(f"Web search failed: {str(e)}")
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "search_depth": self.config.search_depth,
            "max_results": self.config.max_results,
            "include_images": self.config.include_images,
            "include_answer": self.config.include_answer
        })
        return metadata

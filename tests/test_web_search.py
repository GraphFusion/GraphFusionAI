"""
Tests for web search tool.
"""
import pytest
from graphfusionai.tools import WebSearchTool, WebSearchConfig, ToolError

@pytest.fixture
def web_search():
    """Create web search tool instance."""
    config = WebSearchConfig(
        name="web_search",
        description="Search the web using Tavily API",
        api_key="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"
    )
    return WebSearchTool(config=config)

@pytest.mark.asyncio
async def test_web_search_basic(web_search):
    """Test basic web search."""
    results = await web_search.execute(
        "What is GraphFusion AI?",
        search_depth="basic"
    )
    
    assert isinstance(results, dict)
    assert "query" in results
    assert "results" in results
    assert isinstance(results["results"], list)
    assert len(results["results"]) > 0
    
@pytest.mark.asyncio
async def test_web_search_advanced(web_search):
    """Test advanced web search."""
    results = await web_search.execute(
        "Latest developments in graph neural networks",
        search_depth="advanced",
        max_results=10
    )
    
    assert isinstance(results, dict)
    assert len(results["results"]) <= 10
    assert "answer" in results
    
@pytest.mark.asyncio
async def test_web_search_with_images(web_search):
    """Test web search with images."""
    results = await web_search.execute(
        "Show me pictures of graph visualization",
        include_images=True
    )
    
    assert isinstance(results, dict)
    assert any("image" in result for result in results["results"])
    
@pytest.mark.asyncio
async def test_web_search_invalid_query(web_search):
    """Test web search with invalid query."""
    with pytest.raises(ToolError):
        await web_search.execute("")
        
@pytest.mark.asyncio
async def test_web_search_invalid_api_key():
    """Test web search with invalid API key."""
    config = WebSearchConfig(
        name="web_search",
        description="Search the web using Tavily API",
        api_key="invalid-key"
    )
    
    with pytest.raises(ToolError):
        WebSearchTool(config=config)

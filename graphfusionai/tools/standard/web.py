import aiohttp
import asyncio
from typing import Optional, Dict, Any
import time
import json
from urllib.parse import urlparse

from graphfusionai.tools.base import BaseTool, retry
from graphfusionai.tools.registry import ToolRegistry

class WebTool(BaseTool):
    """
    Enhanced tool for web-based actions with advanced features.
    """
    def __init__(self, 
                 concurrent_requests: int = 10,
                 request_timeout: float = 30.0,
                 max_retries: int = 3,
                 cache_ttl: int = 300):
        """
        Initialize WebTool with configurable parameters.
        
        Args:
            concurrent_requests: Maximum concurrent requests
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(
            name="web",
            description="Makes HTTP requests with advanced features and optimizations",
            version="2.0.0",
            async_support=True,
            max_retries=max_retries,
            timeout=request_timeout,
            cache_ttl=cache_ttl
        )
        
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: Optional[aiohttp.ClientSession] = None
        self.domain_timestamps = {}
        self.rate_limits = {}
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
        
    def _validate_url(self, url: str) -> None:
        """Validate URL format and scheme."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Only HTTP(S) URLs are supported")
            
    def _check_rate_limit(self, domain: str) -> None:
        """Check and enforce rate limiting."""
        if domain in self.rate_limits:
            limit = self.rate_limits[domain]
            last_request = self.domain_timestamps.get(domain, 0)
            if time.time() - last_request < limit:
                raise Exception(f"Rate limit exceeded for {domain}")
                
    @retry(max_attempts=3)
    async def _execute(self, 
                      url: str,
                      method: str = "GET",
                      headers: Optional[Dict[str, str]] = None,
                      data: Optional[Dict[str, Any]] = None,
                      params: Optional[Dict[str, str]] = None,
                      verify_ssl: bool = True,
                      follow_redirects: bool = True,
                      timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute web request with advanced features.
        
        Args:
            url: Target URL
            method: HTTP method
            headers: Request headers
            data: Request data/payload
            params: URL parameters
            verify_ssl: Whether to verify SSL certificates
            follow_redirects: Whether to follow redirects
            timeout: Request timeout override
            
        Returns:
            Dictionary containing response data and metadata
        """
        # Validate URL
        self._validate_url(url)
        
        # Get domain for rate limiting
        domain = urlparse(url).netloc
        self._check_rate_limit(domain)
        
        # Update domain timestamp
        self.domain_timestamps[domain] = time.time()
        
        # Prepare request
        method = method.upper()
        headers = headers or {}
        timeout = timeout or self.timeout
        
        # Add default headers
        if 'User-Agent' not in headers:
            headers['User-Agent'] = 'GraphFusionAI-WebTool/2.0'
        
        async with self.semaphore:
            session = await self._get_session()
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                params=params,
                ssl=verify_ssl,
                allow_redirects=follow_redirects,
                timeout=timeout
            ) as response:
                # Raise for bad status
                response.raise_for_status()
                
                # Get response data
                try:
                    content = await response.json()
                    content_type = 'json'
                except:
                    content = await response.text()
                    content_type = 'text'
                
                # Return response with metadata
                return {
                    'url': str(response.url),
                    'status': response.status,
                    'headers': dict(response.headers),
                    'content_type': content_type,
                    'content': content,
                    'elapsed': response.elapsed.total_seconds()
                }
                
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            
    def set_rate_limit(self, domain: str, seconds: float) -> None:
        """Set rate limit for a domain."""
        self.rate_limits[domain] = seconds
        
    def clear_rate_limits(self) -> None:
        """Clear all rate limits."""
        self.rate_limits.clear()
        self.domain_timestamps.clear()

# Register the tool
registry = ToolRegistry()
registry.register(
    WebTool,
    category="networking",
    dependencies=[]
)

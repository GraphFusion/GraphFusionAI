"""
Base tool implementation with enhanced features.
"""
from typing import Any, Dict, Optional, List, Callable, Union, TypeVar
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

T = TypeVar('T', bound='BaseTool')

class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds timeout."""
    pass

class ToolValidationError(ToolError):
    """Raised when tool input validation fails."""
    pass

class ToolMetrics:
    """Tracks performance metrics for tools."""
    
    def __init__(self):
        self.total_calls = 0
        self.total_errors = 0
        self.total_duration = 0
        self.avg_duration = 0
        self.last_used = None
        self.error_types = {}
        self.success_rate = 1.0

    def update(self, duration: float, success: bool, error_type: str = None):
        """Update metrics with new execution data."""
        self.total_calls += 1
        self.last_used = time.time()
        
        if not success:
            self.total_errors += 1
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
            
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_calls
        self.success_rate = (self.total_calls - self.total_errors) / self.total_calls

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for automatic retry on failure."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(self, *args, **kwargs)
                    else:
                        result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator

class ToolConfig(BaseModel):
    """Configuration model for tools."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    async_support: bool = Field(default=False, description="Whether tool supports async execution")
    max_retries: int = Field(default=3, description="Maximum retry attempts on failure")
    timeout: float = Field(default=30.0, description="Execution timeout in seconds")
    cache_ttl: int = Field(default=300, description="Cache time-to-live in seconds")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")

class BaseTool(ABC):
    """Enhanced base class for tools with advanced features and metrics."""

    def __init__(self, config: Optional[ToolConfig] = None, **kwargs):
        """
        Initialize a tool with enhanced configuration.

        Args:
            config: Tool configuration
            **kwargs: Override configuration parameters
        """
        # Initialize configuration
        if config is None:
            config = ToolConfig(**kwargs)
        self.config = config
        
        # Initialize components
        self.metrics = ToolMetrics()
        self.cache = {}
        self.cache_timestamps = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.validators: List[Callable] = []
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"tool.{config.name}")

    @retry()
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Enhanced execute method with metrics, caching, and error handling.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
            
        Raises:
            ToolError: Base class for tool-related errors
            ToolTimeoutError: When execution exceeds timeout
            ToolValidationError: When input validation fails
        """
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            # Check cache
            cache_key = self._get_cache_key(args, kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Validate inputs
            for validator in self.validators:
                if not validator(*args, **kwargs):
                    raise ToolValidationError("Input validation failed")
            
            # Preprocess inputs
            for preprocessor in self.preprocessors:
                args, kwargs = preprocessor(*args, **kwargs)
            
            # Execute with timeout
            if self.config.async_support:
                result = await asyncio.wait_for(
                    self._execute(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._execute,
                    *args,
                    **kwargs
                )
            
            # Postprocess result
            for postprocessor in self.postprocessors:
                result = postprocessor(result)
            
            # Cache result
            self._add_to_cache(cache_key, result)
            
            success = True
            return result
            
        except asyncio.TimeoutError:
            error_type = "timeout"
            raise ToolTimeoutError(f"Execution exceeded {self.config.timeout} seconds")
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.update(duration, success, error_type)

    @abstractmethod
    def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Internal execution method to be implemented by subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        pass

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from arguments."""
        return str(hash((str(args), str(sorted(kwargs.items())))))

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache if valid."""
        if key in self.cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and time.time() - timestamp <= self.config.cache_ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.cache_timestamps[key]
        return None

    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add result to cache."""
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    def add_validator(self, validator: Callable) -> None:
        """Add input validation function."""
        self.validators.append(validator)

    def add_preprocessor(self, preprocessor: Callable) -> None:
        """Add preprocessing function."""
        self.preprocessors.append(preprocessor)

    def add_postprocessor(self, postprocessor: Callable) -> None:
        """Add postprocessing function."""
        self.postprocessors.append(postprocessor)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get enhanced tool metadata.
        
        Returns:
            Dict containing tool metadata and metrics
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "async_support": self.config.async_support,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_rate": self.metrics.success_rate,
                "avg_duration": self.metrics.avg_duration,
                "error_types": self.metrics.error_types
            }
        }

    def clear_cache(self) -> None:
        """Clear the tool's cache."""
        self.cache.clear()
        self.cache_timestamps.clear()

    async def __aenter__(self) -> 'BaseTool':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.executor.shutdown(wait=False)

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        self.executor.shutdown(wait=False)
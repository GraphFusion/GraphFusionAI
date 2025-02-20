from typing import Any, Dict, Optional, List, Callable
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

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

class BaseTool:
    """
    Enhanced base class for tools with advanced features and metrics.
    """
    
    def __init__(self, name: str, description: str, version: str = "1.0.0",
                 async_support: bool = False, max_retries: int = 3,
                 timeout: float = 30.0, cache_ttl: int = 300):
        """
        Initialize a tool with enhanced configuration.

        Args:
            name: Tool name
            description: Tool description
            version: Tool version
            async_support: Whether tool supports async execution
            max_retries: Maximum retry attempts on failure
            timeout: Execution timeout in seconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.name = name
        self.description = description
        self.version = version
        self.async_support = async_support
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # Initialize components
        self.metrics = ToolMetrics()
        self.cache = {}
        self.cache_timestamps = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.validators: List[Callable] = []
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"tool.{name}")

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Enhanced execute method with metrics, caching, and error handling.
        """
        start_time = time.time()
        cache_key = self._get_cache_key(args, kwargs)
        
        try:
            # Check cache
            if cached_result := self._get_from_cache(cache_key):
                self.logger.debug(f"Cache hit for {self.name}")
                return cached_result
            
            # Run validators
            for validator in self.validators:
                validator(*args, **kwargs)
            
            # Run preprocessors
            processed_args = args
            processed_kwargs = kwargs
            for preprocessor in self.preprocessors:
                processed_args, processed_kwargs = preprocessor(*processed_args, **processed_kwargs)
            
            # Execute with timeout
            if self.async_support:
                result = await asyncio.wait_for(
                    self._execute(*processed_args, **processed_kwargs),
                    timeout=self.timeout
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._execute,
                    *processed_args,
                    **processed_kwargs
                )
            
            # Run postprocessors
            for postprocessor in self.postprocessors:
                result = postprocessor(result)
            
            # Update cache
            self._add_to_cache(cache_key, result)
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.update(duration, True)
            
            return result
            
        except Exception as e:
            # Update error metrics
            duration = time.time() - start_time
            self.metrics.update(duration, False, type(e).__name__)
            
            # Log error
            self.logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            raise

    def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Internal execution method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the '_execute' method.")

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from arguments."""
        return f"{self.name}:{hash((args, frozenset(kwargs.items())))}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache if valid."""
        if key in self.cache:
            timestamp = self.cache_timestamps[key]
            if time.time() - timestamp <= self.cache_ttl:
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
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "async_support": self.async_support,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_rate": self.metrics.success_rate,
                "avg_duration": self.metrics.avg_duration,
                "last_used": self.metrics.last_used
            },
            "configuration": {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "cache_ttl": self.cache_ttl
            }
        }

    def clear_cache(self) -> None:
        """Clear the tool's cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
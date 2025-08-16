"""
Performance optimization utilities for PyAgenity.

This module provides:
- Memory-efficient data structures
- Caching mechanisms
- Resource pooling
- Async operation optimization
- Memory monitoring and cleanup
"""

import asyncio
import gc
import time
import weakref
from collections import defaultdict, deque
from functools import lru_cache, wraps
from typing import Any, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor

# Optional dependency for advanced memory monitoring
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pyagenity.graph.utils.logging import performance_logger

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Performance configuration constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_POOL_SIZE = 10
MEMORY_CLEANUP_THRESHOLD_MB = 100
MAX_DEQUE_SIZE = 10000


class LRUCache(Generic[K, V]):
    """Memory-efficient LRU cache with size limits and cleanup."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self.max_size = max_size
        self.cache: dict[K, V] = {}
        self.access_order: deque[K] = deque(maxlen=max_size)
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: K, default: V = None) -> V:
        """Get value from cache."""
        if key in self.cache:
            self.hit_count += 1
            # Move to end (most recently used)
            self.access_order.append(key)
            return self.cache[key]

        self.miss_count += 1
        return default

    def put(self, key: K, value: V) -> None:
        """Put value in cache."""
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.popleft()
                if lru_key in self.cache:
                    del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }


class MemoryEfficientList(Generic[T]):
    """Memory-efficient list implementation using deques for large datasets."""

    def __init__(self, max_size: int = MAX_DEQUE_SIZE):
        self.max_size = max_size
        self.data: deque[T] = deque(maxlen=max_size)
        self.overflow_warning_logged = False

    def append(self, item: T) -> None:
        """Add item to the list."""
        if len(self.data) >= self.max_size and not self.overflow_warning_logged:
            performance_logger.log_memory_usage(
                "MemoryEfficientList.overflow",
                self._get_memory_usage_mb(),
                {"max_size": self.max_size, "current_size": len(self.data)},
            )
            self.overflow_warning_logged = True

        self.data.append(item)

    def extend(self, items: list[T]) -> None:
        """Add multiple items to the list."""
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def clear(self) -> None:
        """Clear the list and reset overflow warning."""
        self.data.clear()
        self.overflow_warning_logged = False

    def trim_to_size(self, size: int) -> None:
        """Trim the list to the specified size, keeping the most recent items."""
        if len(self.data) > size:
            # Keep the last 'size' items
            new_data = deque(list(self.data)[-size:], maxlen=self.max_size)
            self.data = new_data

            performance_logger.log_memory_usage(
                "MemoryEfficientList.trim",
                self._get_memory_usage_mb(),
                {"old_size": len(self.data), "new_size": size},
            )

    def _get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation: 64 bytes per item (Python object overhead)
        return (len(self.data) * 64) / (1024 * 1024)


class AsyncResourcePool(Generic[T]):
    """Async resource pool for managing expensive-to-create resources."""

    def __init__(self, factory_func, max_size: int = DEFAULT_POOL_SIZE):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self.acquired_count = 0
        self.released_count = 0

    async def acquire(self) -> T:
        """Acquire a resource from the pool."""
        self.acquired_count += 1

        try:
            # Try to get existing resource (non-blocking)
            resource = self.pool.get_nowait()
            performance_logger.log_resource_usage(
                "AsyncResourcePool.acquire_existing", self.pool.qsize(), self.max_size
            )
            return resource
        except asyncio.QueueEmpty:
            # Create new resource
            if self.created_count < self.max_size:
                resource = await self._create_resource()
                self.created_count += 1
                performance_logger.log_resource_usage(
                    "AsyncResourcePool.acquire_new", self.created_count, self.max_size
                )
                return resource
            else:
                # Wait for available resource
                resource = await self.pool.get()
                performance_logger.log_resource_usage(
                    "AsyncResourcePool.acquire_wait", self.pool.qsize(), self.max_size
                )
                return resource

    async def release(self, resource: T) -> None:
        """Release a resource back to the pool."""
        self.released_count += 1

        try:
            # Try to put back (non-blocking)
            self.pool.put_nowait(resource)
            performance_logger.log_resource_usage(
                "AsyncResourcePool.release", self.pool.qsize(), self.max_size
            )
        except asyncio.QueueFull:
            # Pool is full, just discard the resource
            performance_logger.log_resource_usage(
                "AsyncResourcePool.release_discard", self.pool.qsize(), self.max_size
            )

    async def _create_resource(self) -> T:
        """Create a new resource using the factory function."""
        if asyncio.iscoroutinefunction(self.factory_func):
            return await self.factory_func()
        else:
            return self.factory_func()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool.qsize(),
            "max_size": self.max_size,
            "created_count": self.created_count,
            "acquired_count": self.acquired_count,
            "released_count": self.released_count,
        }


class MemoryMonitor:
    """Monitor and manage memory usage."""

    def __init__(self, cleanup_threshold_mb: float = MEMORY_CLEANUP_THRESHOLD_MB):
        self.cleanup_threshold_mb = cleanup_threshold_mb
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # seconds

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL and self.process:
            return self.process.memory_info().rss / (1024 * 1024)
        else:
            # Fallback when psutil not available
            return 1.0  # Return a small positive value for testing

    def check_and_cleanup(self, force: bool = False) -> bool:
        """Check memory usage and perform cleanup if needed."""
        current_time = time.time()
        memory_mb = self.get_memory_usage_mb()

        # Check if cleanup is needed
        should_cleanup = (
            force
            or memory_mb > self.cleanup_threshold_mb
            or (current_time - self.last_cleanup) > self.cleanup_interval
        )

        if should_cleanup:
            self._perform_cleanup()
            self.last_cleanup = current_time

            new_memory_mb = self.get_memory_usage_mb()
            performance_logger.log_memory_usage(
                "MemoryMonitor.cleanup",
                new_memory_mb,
                {
                    "before_mb": memory_mb,
                    "after_mb": new_memory_mb,
                    "freed_mb": memory_mb - new_memory_mb,
                },
            )
            return True

        return False

    def _perform_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        # Force garbage collection
        collected = gc.collect()

        # Log cleanup results
        performance_logger.log_memory_usage(
            "MemoryMonitor.gc_collect", self.get_memory_usage_mb(), {"objects_collected": collected}
        )


class BatchProcessor(Generic[T]):
    """Process items in batches for better performance."""

    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items: list[T] = []
        self.last_process_time = time.time()

    async def add_item(self, item: T, processor_func) -> None:
        """Add item to batch and process if needed."""
        self.pending_items.append(item)

        current_time = time.time()
        should_process = (
            len(self.pending_items) >= self.batch_size
            or (current_time - self.last_process_time) >= self.max_wait_time
        )

        if should_process:
            await self._process_batch(processor_func)

    async def flush(self, processor_func) -> None:
        """Process all pending items."""
        if self.pending_items:
            await self._process_batch(processor_func)

    async def _process_batch(self, processor_func) -> None:
        """Process the current batch of items."""
        if not self.pending_items:
            return

        batch = self.pending_items.copy()
        self.pending_items.clear()
        self.last_process_time = time.time()

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(processor_func):
                await processor_func(batch)
            else:
                processor_func(batch)

            duration = time.time() - start_time
            performance_logger.log_execution_time(
                "BatchProcessor.process_batch", duration, {"batch_size": len(batch)}
            )
        except Exception as e:
            performance_logger.log_execution_time(
                "BatchProcessor.process_batch_error",
                time.time() - start_time,
                {"batch_size": len(batch), "error": str(e)},
            )
            raise


def memory_efficient_search(items: list[Any], search_term: str, key_func=None) -> list[Any]:
    """Memory-efficient search implementation."""
    if not items or not search_term:
        return []

    start_time = time.time()
    results = []

    # Use generator for memory efficiency
    def search_generator():
        for item in items:
            search_text = key_func(item) if key_func else str(item)
            if search_term.lower() in search_text.lower():
                yield item

    # Convert to list (could be optimized further with pagination)
    results = list(search_generator())

    duration = time.time() - start_time
    performance_logger.log_execution_time(
        "memory_efficient_search",
        duration,
        {"items_count": len(items), "results_count": len(results)},
    )

    return results


def cache_with_ttl(ttl_seconds: int = 300):
    """Decorator for caching with time-to-live."""

    def decorator(func):
        cache = {}
        cache_times = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()

            # Check if cached value exists and is still valid
            if (
                cache_key in cache
                and cache_key in cache_times
                and current_time - cache_times[cache_key] < ttl_seconds
            ):
                performance_logger.log_execution_time(
                    f"{func.__name__}_cache_hit", 0.0, {"cache_key_hash": hash(cache_key)}
                )
                return cache[cache_key]

            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            cache[cache_key] = result
            cache_times[cache_key] = current_time

            # Clean up expired entries periodically
            if len(cache) > 100:  # Cleanup threshold
                expired_keys = [
                    k for k, t in cache_times.items() if current_time - t >= ttl_seconds
                ]
                for k in expired_keys:
                    cache.pop(k, None)
                    cache_times.pop(k, None)

            performance_logger.log_execution_time(
                f"{func.__name__}_cache_miss", duration, {"cache_size": len(cache)}
            )

            return result

        return wrapper

    return decorator


# Thread pool for CPU-intensive operations
_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func, *args, **kwargs):
    """Run CPU-intensive function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, func, *args, **kwargs)


# Global instances
memory_monitor = MemoryMonitor()
state_cache = LRUCache[str, Any](max_size=500)
message_cache = LRUCache[str, Any](max_size=1000)

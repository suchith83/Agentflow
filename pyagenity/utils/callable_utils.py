import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


_executor = ThreadPoolExecutor()


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """Detect if a callable is a coroutine function (async def)."""
    return asyncio.iscoroutinefunction(func)


async def call_sync_or_async(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call a function that may be sync or async, returning its result.
    If sync, runs in a thread pool to avoid blocking the event loop.
    Thread pool is shared for efficiency.
    """
    if _is_async_callable(func):
        return await func(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))

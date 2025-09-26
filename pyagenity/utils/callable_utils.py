"""
Utilities for calling sync or async functions in PyAgenity.

This module provides helpers to detect async callables and to invoke
functions that may be synchronous or asynchronous, handling thread pool
execution and awaitables.
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """
    Detect if a callable is a coroutine function (async def).

    Args:
        func (Callable[..., Any]): The function to check.

    Returns:
        bool: True if the function is async, False otherwise.
    """
    return asyncio.iscoroutinefunction(func)


async def call_sync_or_async(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call a function that may be sync or async, returning its result.

    If the function is synchronous, it runs in a thread pool to avoid blocking
    the event loop. If the result is awaitable, it is awaited before returning.

    Args:
        func (Callable[..., Any]): The function to call.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The result of the function call, awaited if necessary.
    """
    if _is_async_callable(func):
        return await func(*args, **kwargs)

    # Call sync function in a thread pool
    result = await asyncio.to_thread(func, *args, **kwargs)
    # If the result is awaitable, await it
    if inspect.isawaitable(result):
        return await result
    return result

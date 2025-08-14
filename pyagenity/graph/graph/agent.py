import asyncio
from typing import Any, Dict, Optional, Callable, Union

from pyagenity.graph.exceptions.node_error import NodeError
from pyagenity.graph.utils.command import Command


class Node:
    """Represents a node in the graph."""

    def __init__(
        self,
        name: str,
        func: Callable,
        defer: bool = False,
        retry_policy: Optional[Dict[str, Any]] = None,
        cache_policy: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.func = func
        self.defer = defer
        self.retry_policy = retry_policy or {}
        self.cache_policy = cache_policy or {}
        self.is_async = asyncio.iscoroutinefunction(func)

    async def execute(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Union[Dict[str, Any], Command]:
        """Execute the node function."""
        try:
            if self.is_async:
                result = await self.func(state, config)
            else:
                result = self.func(state, config)
            return result
        except Exception as e:
            raise NodeError(f"Error in node '{self.name}': {str(e)}") from e

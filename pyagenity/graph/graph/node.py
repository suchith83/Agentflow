import asyncio
from collections.abc import Callable
from typing import Any

from pyagenity.graph.exceptions import NodeError
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Command


class Node:
    """Represents a node in the graph."""

    def __init__(
        self,
        name: str,
        func: Callable,
    ):
        self.name = name
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
        checkpointer: Any | None = None,
        store: Any | None = None,
    ) -> dict[str, Any] | Command:
        """Execute the node function."""
        try:
            if self.is_async:
                result = await self.func(state, config, checkpointer, store)
            else:
                result = self.func(state, config, checkpointer, store)
            return result
        except Exception as e:
            raise NodeError(f"Error in node '{self.name}': {e!s}") from e

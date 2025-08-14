import asyncio
from typing import Any, Dict, Optional, Callable, Union

from pyagenity.graph.exceptions.node_error import NodeError
from pyagenity.graph.state.state import AgentState
from pyagenity.graph.utils.command import Command


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
        config: Dict[str, Any],
        checkpointer: Optional[Any] = None,
        store: Optional[Any] = None,
    ) -> Union[Dict[str, Any], Command]:
        """Execute the node function."""
        try:
            if self.is_async:
                result = await self.func(state, config, checkpointer, store)
            else:
                # TODO: Improve me
                result = self.func(state, config)
            return result
        except Exception as e:
            raise NodeError(f"Error in node '{self.name}': {str(e)}") from e

import logging
from collections.abc import AsyncIterable, Callable
from typing import Any, Union

from injectq import Inject

from pyagenity.graph.utils.invoke_node_handler import InvokeNodeHandler
from pyagenity.graph.utils.stream_node_handler import StreamNodeHandler
from pyagenity.publisher import BasePublisher
from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackManager,
    Message,
)

from .tool_node import ToolNode


logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the graph workflow.

    A Node encapsulates a function or ToolNode that can be executed as part of
    a graph workflow. It handles dependency injection, parameter mapping, and
    execution context management.

    The Node class supports both regular callable functions and ToolNode instances
    for handling tool-based operations. It automatically injects dependencies
    based on function signatures and provides legacy parameter support.

    Attributes:
        name (str): Unique identifier for the node within the graph.
        func (Union[Callable, ToolNode]): The function or ToolNode to execute.

    Example:
        >>> def my_function(state, config):
        ...     return {"result": "processed"}
        >>> node = Node("processor", my_function)
        >>> result = await node.execute(state, config)
    """

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode"],
        publisher: BasePublisher | None = Inject[BasePublisher],
    ):
        """Initialize a new Node instance.

        Args:
            name: Unique identifier for the node within the graph.
            func: The function or ToolNode to execute when this node is called.
                Functions should accept at least 'state' and 'config' parameters.
                ToolNode instances handle tool-based operations.
        """
        logger.debug(
            "Initializing node '%s' with func=%s",
            name,
            getattr(func, "__name__", type(func).__name__),
        )
        self.name = name
        self.func = func
        self.publisher = publisher
        self.invoke_handler = InvokeNodeHandler(
            name,
            func,
        )

        self.stream_handler = StreamNodeHandler(
            name,
            func,
        )

    async def execute(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> dict[str, Any] | list[Message]:
        """Execute the node function with dependency injection support and callback hooks."""
        return await self.invoke_handler.invoke(
            config,
            state,
            callback_mgr,
        )

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncIterable[dict[str, Any] | Message]:
        """Stream the node function with dependency injection support and callback hooks."""
        result = self.stream_handler.stream(
            config,
            state,
            callback_mgr,
        )

        async for item in result:
            yield item

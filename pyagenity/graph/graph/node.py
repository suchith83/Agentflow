import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Union

from pyagenity.graph.exceptions import NodeError
from pyagenity.graph.utils import Command
from pyagenity.graph.utils.message import Message

from .tool_node import ToolNode


if TYPE_CHECKING:
    from pyagenity.graph.checkpointer import BaseCheckpointer, BaseStore
    from pyagenity.graph.state import AgentState


class Node:
    """Represents a node in the graph."""

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode"],
    ):
        self.name = name
        self.func = func
        # If func is a ToolNode (registry), it's not a coroutine function
        # Avoid circular import at runtime; use TYPE_CHECKING for annotations

        is_tool = isinstance(func, ToolNode)
        self.is_async = False if is_tool else asyncio.iscoroutinefunction(func)

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None" = None,
        store: "BaseStore | None" = None,
    ) -> Message:
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            # Execute the first tool call for now
            tool_call = last_message.tools_calls[0]
            function_name = tool_call.get("function", {}).get("name", "")
            function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            tool_call_id = tool_call.get("id", "")

            try:
                # Execute the tool function with injectable parameters
                tool_result = await self.func.execute(  # type: ignore
                    function_name,
                    function_args,
                    tool_call_id=tool_call_id,
                    state=state,
                    checkpointer=checkpointer,
                    store=store,
                    config=config,
                )

                # Handle different return types
                if isinstance(tool_result, Message):
                    result = tool_result
                elif isinstance(tool_result, str):
                    # Convert string result to tool message with tool_call_id
                    result = Message.tool_message(tool_call_id=tool_call_id, content=tool_result)
                else:
                    # Convert other types to string then to tool message
                    result = Message.tool_message(
                        tool_call_id=tool_call_id, content=str(tool_result)
                    )
            except Exception as e:
                # Return error message
                result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=f"Error executing tool: {e}",
                    is_error=True,
                )
        else:
            # No tool calls to execute, return available tools
            raise NodeError("No tool calls to execute")

        return result

    async def execute(
        self,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None" = None,
        store: "BaseStore | None" = None,
    ) -> dict[str, Any] | Command:
        """Execute the node function."""
        try:
            if isinstance(self.func, ToolNode):
                # Look for tool calls in the last message that need execution
                if state.context and len(state.context) > 0:
                    last_message = state.context[-1]
                    result = await self._call_tools(
                        last_message,
                        state,
                        config,
                        checkpointer=checkpointer,
                        store=store,
                    )
                    # Check if last message has tool calls to execute
                else:
                    # No context, return available tools
                    raise NodeError("No context available for tool execution")

            elif self.is_async:
                result = await self.func(state, config, checkpointer, store)
            else:
                # Execute sync function in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor() as executor:
                    # its callable sync function so no worry
                    result = await loop.run_in_executor(
                        executor,
                        lambda: self.func(
                            state,
                            config,
                            checkpointer,
                            store,
                        ),  # type: ignore
                    )
            return result  # pyright: ignore[reportReturnType]
        except Exception as e:
            raise NodeError(f"Error in node '{self.name}': {e!s}") from e

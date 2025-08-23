import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from pyagenity.exceptions import NodeError
from pyagenity.utils import (
    Command,
    DependencyContainer,
    Message,
    call_sync_or_async,
    get_injectable_param_name,
)

from .tool_node import ToolNode


if TYPE_CHECKING:
    from pyagenity.checkpointer import BaseCheckpointer
    from pyagenity.state import AgentState
    from pyagenity.store import BaseStore


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
    ):
        """Initialize a new Node instance.

        Args:
            name: Unique identifier for the node within the graph.
            func: The function or ToolNode to execute when this node is called.
                Functions should accept at least 'state' and 'config' parameters.
                ToolNode instances handle tool-based operations.
        """
        self.name = name
        self.func = func

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None" = None,
        store: "BaseStore | None" = None,
        dependency_container: DependencyContainer | None = None,
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
            meta = {"function_name": function_name, "function_argument": function_args}

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
                    dependency_container=dependency_container,
                )

                # TODO: Allow state also

                # Handle different return types
                if isinstance(tool_result, Message):
                    # lets update the meta
                    meta_data = tool_result.metadata or {}
                    meta.update(meta_data)
                    tool_result.metadata = meta
                    result = tool_result
                elif isinstance(tool_result, str):
                    # Convert string result to tool message with tool_call_id
                    result = Message.tool_message(
                        tool_call_id=tool_call_id, content=tool_result, meta=meta
                    )
                else:
                    # Convert other types to string then to tool message
                    result = Message.tool_message(
                        tool_call_id=tool_call_id, content=str(tool_result), meta=meta
                    )
            except Exception as e:
                # Return error message
                result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=f"Error executing tool: {e}",
                    is_error=True,
                    meta=meta,
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
        dependency_container: DependencyContainer | None = None,
    ) -> dict[str, Any] | Command:
        """Execute the node function with dependency injection support."""
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
                        dependency_container=dependency_container,
                    )
                    # Check if last message has tool calls to execute
                else:
                    # No context, return available tools
                    raise NodeError("No context available for tool execution")

            else:
                # Inject dependencies based on function signature
                kwargs = self._prepare_function_kwargs(
                    state, config, checkpointer, store, dependency_container
                )
                result = await call_sync_or_async(self.func, **kwargs)
            return result  # pyright: ignore[reportReturnType]
        except Exception as e:
            print("Error occurred while executing node function", e)
            raise NodeError(f"Error in node '{self.name}': {e!s}") from e

    def _prepare_function_kwargs(
        self,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None",
        store: "BaseStore | None",
        dependency_container: DependencyContainer | None,
    ) -> dict[str, Any]:
        """Prepare keyword arguments for function execution with dependency injection."""
        if isinstance(self.func, ToolNode):
            return {}  # ToolNode has its own parameter handling

        kwargs = {}

        # Get function signature
        sig = inspect.signature(self.func)

        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Try injectable parameter mapping first
            injectable_value = self._get_injectable_value(
                param, param_name, state, config, checkpointer, store, dependency_container
            )
            if injectable_value is not None:
                kwargs[param_name] = injectable_value
            elif self._is_legacy_parameter(param_name):
                # Handle legacy parameters by name
                kwargs[param_name] = self._get_legacy_value(
                    param_name, state, config, checkpointer, store
                )
            elif (
                param.default == param.empty
                and param_name not in ["self", "cls"]
                and dependency_container
                and dependency_container.has(param_name)
            ):
                # Try dependency container for untyped required params
                kwargs[param_name] = dependency_container.get(param_name)

        return kwargs

    def _get_injectable_value(
        self,
        param,
        param_name: str,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None",
        store: "BaseStore | None",
        dependency_container: DependencyContainer | None,
    ) -> Any | None:
        """Get value for injectable parameter, returns None if not injectable."""
        if not param.annotation or param.annotation == param.empty:
            return None

        injectable_param = get_injectable_param_name(param.annotation)
        if not injectable_param:
            return None

        # Map injectable parameters to their values
        injectable_values = {
            "state": state,
            "config": config,
            "checkpointer": checkpointer,
            "store": store,
        }

        if injectable_param in injectable_values:
            return injectable_values[injectable_param]

        if injectable_param == "dependency":
            return self._handle_dependency_injection(param, param_name, dependency_container)

        return None

    def _handle_dependency_injection(
        self, param, param_name: str, dependency_container: DependencyContainer | None
    ) -> Any | None:
        """Handle dependency injection for InjectDep parameters."""
        if dependency_container and dependency_container.has(param_name):
            return dependency_container.get(param_name)
        if param.default == param.empty:
            raise NodeError(f"Required dependency '{param_name}' not found in container")
        return None  # Use default

    def _is_legacy_parameter(self, param_name: str) -> bool:
        """Check if parameter is a legacy parameter that should be injected by name."""
        return param_name in ["state", "config", "checkpointer", "store"]

    def _get_legacy_value(
        self,
        param_name: str,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None",
        store: "BaseStore | None",
    ) -> Any:
        """Get value for legacy parameter by name.

        Args:
            param_name: Name of the parameter to get value for.
            state: Current agent state.
            config: Configuration dictionary.
            checkpointer: Optional checkpointer instance.
            store: Optional store instance.

        Returns:
            The value for the requested parameter, or None if not found.
        """
        legacy_values = {
            "state": state,
            "config": config,
            "checkpointer": checkpointer,
            "store": store,
        }
        return legacy_values.get(param_name)

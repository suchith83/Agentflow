import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from pyagenity.graph.exceptions import NodeError
from pyagenity.graph.utils import Command
from pyagenity.graph.utils.callable_utils import call_sync_or_async
from pyagenity.graph.utils.dependency_injection import DependencyContainer
from pyagenity.graph.utils.injectable import get_injectable_param_name
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

        if injectable_param == "state":
            return state
        elif injectable_param == "config":
            return config
        elif injectable_param == "checkpointer":
            return checkpointer
        elif injectable_param == "store":
            return store
        elif injectable_param == "dependency":
            # For InjectDep, use the parameter name to look up the dependency
            if dependency_container and dependency_container.has(param_name):
                return dependency_container.get(param_name)
            elif param.default == param.empty:
                # Required dependency not found
                raise NodeError(f"Required dependency '{param_name}' not found in container")
            # If default exists and dependency not found, don't inject (use default)

        return None

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
        """Get value for legacy parameter by name."""
        if param_name == "state":
            return state
        elif param_name == "config":
            return config
        elif param_name == "checkpointer":
            return checkpointer
        elif param_name == "store":
            return store
        return None

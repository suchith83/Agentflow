import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from pyagenity.exceptions import NodeError
from pyagenity.publisher import Event, EventType, SourceType
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    Command,
    DependencyContainer,
    InvocationType,
    Message,
    call_sync_or_async,
    default_callback_manager,
    get_injectable_param_name,
)

from .tool_node import ToolNode


if TYPE_CHECKING:
    from pyagenity.checkpointer import BaseCheckpointer
    from pyagenity.publisher import BasePublisher
    from pyagenity.state import AgentState
    from pyagenity.store import BaseStore

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
        publisher: "BasePublisher | None" = None,
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
        self.event = Event(
            source=SourceType.NODE,
            event_type=EventType.INITIALIZE,
        )

    async def _publish_event(
        self,
        event: Event,
    ) -> None:
        """Publish an event if publisher is configured."""
        if self.publisher:
            try:
                await self.publisher.publish(event)
                logger.debug("Published event: %s", event)
            except Exception as e:
                logger.error("Failed to publish event: %s", e)

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None" = None,
        store: "BaseStore | None" = None,
        dependency_container: DependencyContainer | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> Message:
        logger.debug("Node '%s' calling tools from message", self.name)
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

            logger.info(
                "Node '%s' executing tool '%s' with %d arguments",
                self.name,
                function_name,
                len(function_args),
            )
            logger.debug("Tool arguments: %s", function_args)

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
                    callback_manager=callback_manager,
                )
                logger.debug("Node '%s' tool execution completed successfully", self.name)

                # TODO: Allow state also

                # Handle different return types
                if isinstance(tool_result, Message):
                    logger.debug("Node '%s' tool execution returned a Message", self.name)
                    # lets update the meta
                    meta_data = tool_result.metadata or {}
                    meta.update(meta_data)
                    tool_result.metadata = meta
                    result = tool_result
                elif isinstance(tool_result, str):
                    logger.debug("Node '%s' tool execution returned a string", self.name)
                    # Convert string result to tool message with tool_call_id
                    result = Message.tool_message(
                        tool_call_id=tool_call_id, content=tool_result, meta=meta
                    )
                else:
                    # Convert other types to string then to tool message
                    logger.debug(
                        "Node '%s' tool execution returned an unexpected type: %s",
                        self.name,
                        type(tool_result),
                    )
                    result = Message.tool_message(
                        tool_call_id=tool_call_id, content=str(tool_result), meta=meta
                    )
            except Exception as e:
                # Return error message
                logger.exception("Node '%s' tool execution failed: %s", self.name, e)
                result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=f"Error executing tool: {e}",
                    is_error=True,
                    meta=meta,
                )
        else:
            # No tool calls to execute, return available tools
            logger.exception("Node '%s': No tool calls to execute", self.name)
            raise NodeError("No tool calls to execute")

        return result

    async def execute(
        self,
        state: "AgentState",
        config: dict[str, Any],
        checkpointer: "BaseCheckpointer | None" = None,
        store: "BaseStore | None" = None,
        dependency_container: DependencyContainer | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> dict[str, Any] | Command:
        """Execute the node function with dependency injection support and callback hooks."""
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

        callback_mgr = callback_manager or default_callback_manager

        await self._publish_event(
            Event(
                source=SourceType.NODE,
                event_type=EventType.RUNNING,
                payload={
                    "state": state.model_dump(
                        exclude={"execution_meta"},
                        exclude_none=True,
                    ),
                    "config": config,
                },
            )
        )

        try:
            if isinstance(self.func, ToolNode):
                logger.debug("Node '%s' is a ToolNode, executing tool calls", self.name)
                # This is tool execution - handled separately in ToolNode
                if state.context and len(state.context) > 0:
                    last_message = state.context[-1]
                    logger.debug("Node '%s' processing tool calls from last message", self.name)
                    result = await self._call_tools(
                        last_message,
                        state,
                        config,
                        checkpointer=checkpointer,
                        store=store,
                        dependency_container=dependency_container,
                        callback_manager=callback_manager,
                    )
                    # Check if last message has tool calls to execute
                else:
                    # No context, return available tools
                    error_msg = "No context available for tool execution"
                    logger.error("Node '%s': %s", self.name, error_msg)
                    raise NodeError(error_msg)

            else:
                logger.debug("Node '%s' is a regular function, executing with callbacks", self.name)
                # This is a regular function - likely AI function
                # Create callback context for AI invocation
                context = CallbackContext(
                    invocation_type=InvocationType.AI,
                    node_name=self.name,
                    function_name=getattr(self.func, "__name__", str(self.func)),
                    metadata={"config": config},
                )

                # Prepare function arguments
                kwargs = self._prepare_function_kwargs(
                    state, config, checkpointer, store, dependency_container
                )
                logger.debug(
                    "Node '%s' prepared %d function arguments: %s",
                    self.name,
                    len(kwargs),
                    list(kwargs.keys()),
                )

                # Execute before_invoke callbacks
                input_data = {"state": state, "kwargs": kwargs, "config": config}

                try:
                    logger.debug("Node '%s' executing before_invoke callbacks", self.name)
                    # Execute before_invoke callbacks
                    input_data = await callback_mgr.execute_before_invoke(context, input_data)

                    # Extract potentially modified data
                    modified_state = input_data.get("state", state)
                    modified_kwargs = input_data.get("kwargs", kwargs)
                    modified_config = input_data.get("config", config)

                    # Update kwargs with potentially modified state and config
                    if "state" in modified_kwargs:
                        modified_kwargs["state"] = modified_state
                    if "config" in modified_kwargs:
                        modified_kwargs["config"] = modified_config

                    logger.debug("Node '%s' executing function", self.name)
                    # Execute the actual function
                    result = await call_sync_or_async(self.func, **modified_kwargs)
                    logger.debug("Node '%s' function execution completed", self.name)

                    logger.debug("Node '%s' executing after_invoke callbacks", self.name)
                    # Execute after_invoke callbacks
                    result = await callback_mgr.execute_after_invoke(context, input_data, result)

                except Exception as e:
                    logger.warning(
                        "Node '%s' execution failed, executing error callbacks: %s", self.name, e
                    )
                    # Execute error callbacks
                    recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

                    if recovery_result is not None:
                        logger.info(
                            "Node '%s' recovered from error using callback result",
                            self.name,
                        )
                        # Use recovery result instead of raising the error
                        result = recovery_result
                    else:
                        # Re-raise the original error
                        logger.error("Node '%s' could not recover from error", self.name)
                        raise

            logger.info("Node '%s' execution completed successfully", self.name)
            return result  # pyright: ignore[reportReturnType]
        except Exception as e:
            # This is the final catch-all for node execution errors
            logger.exception("Node '%s' execution failed: %s", self.name, e)
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
            logger.debug(
                "Node '%s' is a ToolNode, using tool-specific parameter handling", self.name
            )
            return {}  # ToolNode has its own parameter handling

        kwargs = {}

        # Get function signature
        sig = inspect.signature(self.func)
        logger.debug("Node '%s' function signature: %s", self.name, sig)

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

        logger.debug("Node '%s' prepared function kwargs: %s", self.name, kwargs)
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
            logger.debug("Node '%s' parameter '%s' is not injectable", self.name, param_name)
            return None

        injectable_param = get_injectable_param_name(param.annotation)
        if not injectable_param:
            logger.debug("Node '%s' parameter '%s' is not injectable", self.name, param_name)
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
            logger.debug("Node '%s' parameter '%s' is a dependency", self.name, param_name)
            return self._handle_dependency_injection(param, param_name, dependency_container)

        return None

    def _handle_dependency_injection(
        self, param, param_name: str, dependency_container: DependencyContainer | None
    ) -> Any | None:
        """Handle dependency injection for InjectDep parameters."""
        if dependency_container and dependency_container.has(param_name):
            return dependency_container.get(param_name)
        if param.default == param.empty:
            logger.debug("Node '%s' parameter '%s' is a required dependency", self.name, param_name)
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

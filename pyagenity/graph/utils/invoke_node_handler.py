import inspect
import json
import logging
from collections.abc import Callable
from typing import Any, Union

from injectq import Inject

from pyagenity.exceptions import NodeError
from pyagenity.graph.tool_node import ToolNode
from pyagenity.publisher import BasePublisher, Event, EventType, SourceType
from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    Command,
    InvocationType,
    Message,
    call_sync_or_async,
)


logger = logging.getLogger(__name__)


class InvokeNodeHandler:
    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode"],
        publisher: BasePublisher | None = Inject[BasePublisher],
    ):
        self.name = name
        self.func = func
        self.publisher = publisher

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

    async def _handle_single_tool(
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
    ) -> Message:
        function_name = tool_call.get("function", {}).get("name", "")
        function_args: dict = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        tool_call_id = tool_call.get("id", "")

        logger.info(
            "Node '%s' executing tool '%s' with %d arguments",
            self.name,
            function_name,
            len(function_args),
        )
        logger.debug("Tool arguments: %s", function_args)

        # Execute the tool function with injectable parameters
        tool_result = await self.func.invoke(  # type: ignore
            function_name,  # type: ignore
            function_args,
            tool_call_id=tool_call_id,
            state=state,
            config=config,
        )
        logger.debug("Node '%s' tool execution completed successfully", self.name)

        return tool_result

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
    ) -> list[Message]:
        logger.debug("Node '%s' calling tools from message", self.name)
        result: list[Message] = []
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            # Execute the first tool call for now
            tool_call = last_message.tools_calls[0]
            for tool_call in last_message.tools_calls:
                res = await self._handle_single_tool(
                    tool_call,
                    state,
                    config,
                )
                result.append(res)
        else:
            # No tool calls to execute, return available tools
            logger.exception("Node '%s': No tool calls to execute", self.name)
            raise NodeError("No tool calls to execute")

        return result

    def _prepare_input_data(
        self,
        state: "AgentState",
        config: dict[str, Any],
    ) -> dict:
        sig = inspect.signature(self.func)  # type: ignore Tool node won't come here
        input_data = {}
        default_data = {
            "state": state,
            "config": config,
        }

        # # Get injectable parameters to determine which ones to exclude from manual passing
        # # Prepare function arguments (excluding injectable parameters)
        for param_name, param in sig.parameters.items():
            # Skip *args/**kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # check its state, config
            if param_name in ["state", "config"]:
                input_data[param_name] = default_data[param_name]
            # Include regular function arguments
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Missing required parameter '{param_name}' for function '{self.func}'"
                )

        return input_data

    async def _call_normal_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any]:
        logger.debug("Node '%s' calling normal function", self.name)
        result: dict[str, Any] = {}

        logger.debug("Node '%s' is a regular function, executing with callbacks", self.name)
        # This is a regular function - likely AI function
        # Create callback context for AI invocation
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=getattr(self.func, "__name__", str(self.func)),
            metadata={"config": config},
        )

        # Execute before_invoke callbacks
        input_data = self._prepare_input_data(
            state,
            config,
        )

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)
            # Execute the actual function
            result = await call_sync_or_async(
                self.func,  # type: ignore
                **input_data,
            )
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

        return result

    async def invoke(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> dict[str, Any] | Command:
        """Execute the node function with dependency injection support and callback hooks."""
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

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
                    )
                    # Check if last message has tool calls to execute
                else:
                    # No context, return available tools
                    error_msg = "No context available for tool execution"
                    logger.error("Node '%s': %s", self.name, error_msg)
                    raise NodeError(error_msg)

            else:
                result = await self._call_normal_node(
                    state,
                    config,
                    callback_mgr,
                )

            logger.info("Node '%s' execution completed successfully", self.name)
            return result  # pyright: ignore[reportReturnType]
        except Exception as e:
            # This is the final catch-all for node execution errors
            logger.exception("Node '%s' execution failed: %s", self.name, e)
            raise NodeError(f"Error in node '{self.name}': {e!s}") from e

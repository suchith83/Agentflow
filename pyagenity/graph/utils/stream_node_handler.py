import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import Any, Union

from injectq import Inject

from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter
from pyagenity.exceptions import NodeError
from pyagenity.graph.tool_node import ToolNode
from pyagenity.graph.utils.stream_utils import check_non_streaming
from pyagenity.graph.utils.utils import process_node_result
from pyagenity.publisher.events import ContentType, Event, EventModel, EventType
from pyagenity.publisher.publish import publish_event
from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    Message,
    call_sync_or_async,
)
from pyagenity.utils.command import Command

from .handler_mixins import BaseLoggingMixin


logger = logging.getLogger(__name__)


class StreamNodeHandler(BaseLoggingMixin):
    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode"],
    ):
        self.name = name
        self.func = func

    async def _handle_single_tool(
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
    ) -> AsyncIterable[Message]:
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
        tool_result_gen = self.func.stream(  # type: ignore
            function_name,  # type: ignore
            function_args,
            tool_call_id=tool_call_id,
            state=state,
            config=config,
        )
        logger.debug("Node '%s' tool execution completed successfully", self.name)

        async for result in tool_result_gen:
            if isinstance(result, Message):
                yield result

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
    ) -> AsyncIterable[Message]:
        logger.debug("Node '%s' calling tools from message", self.name)
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            # Execute tool calls
            for tool_call in last_message.tools_calls:
                result_gen = self._handle_single_tool(
                    tool_call,
                    state,
                    config,
                )
                async for result in result_gen:
                    if isinstance(result, Message):
                        yield result
        else:
            # No tool calls to execute, return available tools
            logger.exception("Node '%s': No tool calls to execute", self.name)
            raise NodeError("No tool calls to execute")

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

    async def _call_normal_node(  # noqa: PLR0912, PLR0915
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> AsyncIterable[dict[str, Any] | Message]:
        logger.debug("Node '%s' calling normal function", self.name)
        result: dict[str, Any] | Message = {}

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

        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={
                "node": self.name,
                "function_name": getattr(self.func, "__name__", str(self.func)),
                "last_message": last_message.model_dump() if last_message else None,
            },
        )
        publish_event(event)

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)
            event.event_type = EventType.PROGRESS
            event.content = "Function execution started"
            publish_event(event)

            # Execute the actual function
            result = await call_sync_or_async(
                self.func,  # type: ignore
                **input_data,
            )
            logger.debug("Node '%s' function execution completed", self.name)

            logger.debug("Node '%s' executing after_invoke callbacks", self.name)
            # Execute after_invoke callbacks
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            # Now lets convert the response here only, upstream will be easy to handle
            ##############################################################################
            ################### Logics for streaming ##########################
            ##############################################################################
            """
            Check user sending command or not
            if command then we will check its streaming or not
            if streaming then we will yield from converter stream
            if not streaming then we will convert it and yield end event
            if its not command then we will check its streaming or not
            if streaming then we will yield from converter stream
            if not streaming then we will convert it and yield end event
            """
            # first check its sync and not streaming
            next_node = None
            final_result = result
            # if type of command then we will update it
            if isinstance(result, Command):
                # now check the updated
                if result.update:
                    final_result = result.update

                if result.state:
                    state = result.state
                    for msg in state.context:
                        yield msg

                next_node = result.goto

            messages = []
            if check_non_streaming(final_result):
                new_state, messages, next_node = await process_node_result(
                    final_result,
                    state,
                    messages,
                )
                event.data["state"] = new_state.model_dump()
                event.event_type = EventType.END
                event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                event.data["next_node"] = next_node
                publish_event(event)
                for m in messages:
                    yield m

                yield {
                    "is_non_streaming": True,
                    "state": new_state,
                    "messages": messages,
                    "next_node": next_node,
                }
                return  # done

            # If the result is a ConverterCall with stream=True, use the converter
            if isinstance(result, ModelResponseConverter) and result.response:
                stream_gen = result.stream(
                    config,
                    node_name=self.name,
                    meta={
                        "function_name": getattr(self.func, "__name__", str(self.func)),
                    },
                )
                # this will return event_model or message
                async for item in stream_gen:
                    if isinstance(item, Message) and not item.delta:
                        messages.append(item)
                    yield item
            # Things are done, so publish event and yield final response
            event.event_type = EventType.END
            if messages:
                final_msg = messages[-1]
                event.data["message"] = final_msg.model_dump()
                # Populate simple content and structured blocks when available
                event.content = (
                    final_msg.text() if isinstance(final_msg.content, list) else final_msg.content
                )
                if isinstance(final_msg.content, list):
                    event.content_blocks = final_msg.content
            else:
                event.data["message"] = None
                event.content = ""
                event.content_blocks = None
            event.content_type = [ContentType.MESSAGE, ContentType.STATE]
            publish_event(event)
            # if user use command and its streaming in that case we need to handle next node also
            yield {
                "is_non_streaming": False,
                "messages": messages,
                "next_node": next_node,
            }

        except Exception as e:
            logger.warning(
                "Node '%s' execution failed, executing error callbacks: %s", self.name, e
            )
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                logger.info(
                    "Node '%s' recovered from error using callback result",
                    self.name,
                )
                # Use recovery result instead of raising the error
                event.event_type = EventType.END
                event.content = "Function execution recovered from error"
                event.data["message"] = recovery_result.model_dump()
                event.content_type = [ContentType.MESSAGE, ContentType.STATE]
                publish_event(event)

                yield recovery_result
            else:
                # Re-raise the original error
                logger.error("Node '%s' could not recover from error", self.name)
                event.event_type = EventType.ERROR
                event.content = f"Function execution failed: {e}"
                event.data["error"] = str(e)
                event.content_type = [ContentType.ERROR, ContentType.STATE]
                publish_event(event)
                raise

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncGenerator[dict[str, Any] | Message]:
        """Execute the node function with dependency injection support and callback hooks."""
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

        # In this function publishing events not required
        # If its tool node, its already handled there, from start to end
        # In this class we need to handle normal function calls only
        # We will yield events from here only for normal function calls
        # ToolNode will yield events from its own stream method

        try:
            if isinstance(self.func, ToolNode):
                logger.debug("Node '%s' is a ToolNode, executing tool calls", self.name)
                # This is tool execution - handled separately in ToolNode
                if state.context and len(state.context) > 0:
                    last_message = state.context[-1]
                    logger.debug("Node '%s' processing tool calls from last message", self.name)
                    result = self._call_tools(
                        last_message,
                        state,
                        config,
                    )
                    async for item in result:
                        yield item
                    # Check if last message has tool calls to execute
                else:
                    # No context, return available tools
                    error_msg = "No context available for tool execution"
                    logger.error("Node '%s': %s", self.name, error_msg)
                    raise NodeError(error_msg)

            else:
                result = self._call_normal_node(
                    state,
                    config,
                    callback_mgr,
                )
                async for item in result:
                    yield item

            logger.info("Node '%s' execution completed successfully", self.name)
        except Exception as e:
            # This is the final catch-all for node execution errors
            logger.exception("Node '%s' execution failed: %s", self.name, e)
            raise NodeError(f"Error in node '{self.name}': {e!s}") from e

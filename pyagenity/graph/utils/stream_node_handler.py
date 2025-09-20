import inspect  # isort: skip_file
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import Any, Union

from injectq import Inject
from litellm import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream

from pyagenity.exceptions import NodeError
from pyagenity.graph.tool_node import ToolNode
from pyagenity.graph.utils.stream_utils import check_non_streaming
from pyagenity.graph.utils.utils import process_node_result
from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    Message,
    call_sync_or_async,
)
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

from .handler_mixins import BaseLoggingMixin, EventPublishingMixin


logger = logging.getLogger(__name__)


class StreamNodeHandler(BaseLoggingMixin, EventPublishingMixin):
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
    ) -> AsyncIterable[dict[str, Any] | EventModel | Message]:
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
        self.publish_event(event)
        yield event

        stream_event = EventModel.stream(
            config,
            node_name=self.name,
        )

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)
            event.event_type = EventType.PROGRESS
            event.content = "Function execution started"
            yield event
            self.publish_event(event)

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
            1. First check user sending any streaming or not, we can use AsyncIterable
            If yes, we need to yield from it and collect chunks, we expect user are sending
            chunks[text] in each yield, we will collect all and send as final response and
            save inside message...
            2. As this library has first class support for litellm, we will check if its
            returning CustomStreamWrapper, if yes we will yield from it directly
            3. If its normal response, we will convert to dict and send as normal response
            """
            # first check its sync and not streaming
            messages = []
            if check_non_streaming(result):
                new_state, messages, next_node = process_node_result(result, state, messages)
                event.data["state"] = new_state.model_dump()
                event.event_type = EventType.END
                event.content = "Function execution completed"
                event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                event.data["next_node"] = next_node
                self.publish_event(event)
                yield event

                yield {
                    "is_non_streaming": True,
                    "state": new_state,
                    "messages": messages,
                    "next_node": next_node,
                }
                return  # done

            # Now check its streaming
            # Lets handle litellm streaming first

            if isinstance(result, CustomStreamWrapper):
                accumulated_content = ""
                tool_calls = []
                tool_ids = set()
                accumulated_reasoning_content = ""
                seq = 0
                async for chunk in result:
                    if not chunk:
                        continue

                    msg: ModelResponseStream = chunk  # type: ignore
                    if msg is None:
                        continue
                    if msg.choices is None or len(msg.choices) == 0:
                        continue
                    delta = msg.choices[0].delta
                    if delta is None:
                        continue

                    stream_event.content = delta.content if delta.content else ""
                    stream_event.data = {
                        "reasoning_content": getattr(delta, "reasoning_content", "") or "",
                    }
                    seq += 1
                    stream_event.sequence_id = seq
                    yield stream_event

                    accumulated_content += delta.content if delta.content else ""
                    accumulated_reasoning_content += getattr(delta, "reasoning_content", "") or ""
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if not tc:
                                continue

                            if tc.id in tool_ids:
                                continue

                            tool_ids.add(tc.id)
                            tool_calls.append(tc.model_dump())

                    # yield tool calls as well
                    if tool_calls and len(tool_calls) > 0:
                        seq += 1
                        stream_event.data["tool_calls"] = tool_calls
                        stream_event.content = ""
                        stream_event.delta = True
                        stream_event.sequence_id = seq
                        yield stream_event

                # Loop done, now send final response
                message = Message.create(
                    role="assistant",
                    content=accumulated_content,
                    reasoning=accumulated_reasoning_content,
                    tools_calls=tool_calls,
                    meta={
                        "node": self.name,
                        "function_name": getattr(self.func, "__name__", str(self.func)),
                        "last_message": last_message.model_dump() if last_message else None,
                    },
                )
                messages.append(message)

                stream_event.event_type = EventType.END
                stream_event.content = accumulated_content
                stream_event.sequence_id = seq + 1
                stream_event.delta = False
                stream_event.content_type = [
                    ContentType.MESSAGE,
                    ContentType.REASONING,
                    ContentType.TEXT,
                ]
                stream_event.data = {
                    "state": state.model_dump(),
                    "messages": [m.model_dump() for m in messages] if messages else [],
                    "next_node": None,
                    "reasoning_content": accumulated_reasoning_content,
                    "final_response": accumulated_content,
                    "tool_calls": tool_calls,
                }
                yield stream_event

                logger.info("Node '%s' execution completed successfully", self.name)
                # yield message, that will be collected upstream
                yield message

            # Things are done, so publish event and yield final response
            event.event_type = EventType.END
            event.content = "Function execution completed"
            event.data["message"] = messages[0].model_dump() if messages else None
            event.content_type = [ContentType.MESSAGE, ContentType.STATE]
            self.publish_event(event)
            yield event

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
                self.publish_event(event)
                yield event

                yield recovery_result
            else:
                # Re-raise the original error
                logger.error("Node '%s' could not recover from error", self.name)
                event.event_type = EventType.ERROR
                event.content = f"Function execution failed: {e}"
                event.data["error"] = str(e)
                event.content_type = [ContentType.ERROR, ContentType.STATE]
                self.publish_event(event)
                yield event
                raise

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncGenerator[dict[str, Any] | EventModel | Message]:
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

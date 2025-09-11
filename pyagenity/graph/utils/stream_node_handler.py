import inspect
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
from pyagenity.publisher import BasePublisher, Event, EventType, SourceType
from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    Message,
    call_sync_or_async,
)
from pyagenity.utils.streaming import StreamChunk, StreamEvent


logger = logging.getLogger(__name__)


class StreamNodeHandler:
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

    async def _call_normal_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> AsyncIterable[dict[str, Any] | StreamChunk | Message]:
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

        run_id = config.get("run_id", "")
        cfg = {
            "thread_id": config.get("thread_id", ""),
            "run_id": run_id,
            "run_timestamp": config.get("timestamp", ""),
        }

        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        data = {
            "node": self.name,
            "last_message": last_message.model_dump() if last_message else None,
        }

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)

            yield StreamChunk(
                event=StreamEvent.NODE_EXECUTION,
                event_type="Before",
                run_id=run_id,
                data=data,
                metadata=cfg,
            )

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
                yield StreamChunk(
                    event=StreamEvent.NODE_EXECUTION,
                    event_type="Before",
                    run_id=run_id,
                    data={
                        "streaming": False,
                        "node": self.name,
                        "messages": [m.model_dump() for m in messages] if messages else [],
                    },
                    metadata=cfg,
                )

                yield {
                    "is_non_streaming": True,
                    "state": new_state,
                    "messages": messages,
                    "next_node": next_node,
                }

            # Now check its streaming
            # Lets handle litellm streaming first

            if isinstance(result, CustomStreamWrapper):
                accumulated_content = ""
                tool_calls = []
                tool_ids = set()
                accumulated_reasoning_content = ""
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

                    yield StreamChunk(
                        event=StreamEvent.NODE_EXECUTION,
                        event_type="After",
                        run_id=run_id,
                        data={
                            "node": self.name,
                            "delta": msg.model_dump(),
                        },
                        metadata=cfg,
                    )

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

                # Loop done, now send final response
                message = Message.create(
                    role="assistant",
                    content=accumulated_content,
                    reasoning=accumulated_reasoning_content,
                    tools_calls=tool_calls,
                    meta=cfg,
                )
                yield message

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
                yield recovery_result
            else:
                # Re-raise the original error
                logger.error("Node '%s' could not recover from error", self.name)
                raise

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncGenerator[dict[str, Any] | StreamChunk | Message]:
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

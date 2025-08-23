from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.checkpointer import BaseCheckpointer
from pyagenity.exceptions import GraphRecursionError
from pyagenity.publisher import BasePublisher, Event, EventType
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.store import BaseStore
from pyagenity.utils import (
    END,
    CallbackManager,
    Message,
    ResponseGranularity,
    StreamChunk,
    default_callback_manager,
    extract_content_from_response,
    is_async_streaming_response,
    is_streaming_response,
    simulate_async_streaming,
)

from .utils import (
    call_realtime_sync,
    get_default_event,
    get_next_node,
    load_or_create_state,
    process_node_result,
    sync_data,
)


# Import StateGraph only for typing to avoid circular import at runtime
if TYPE_CHECKING:
    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class StreamHandler[StateT: AgentState]:
    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
        publisher: BasePublisher | None = None,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.debug = debug
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self.callback_manager = callback_manager or default_callback_manager
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

    async def _check_interrupted(
        self,
        state: StateT,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if state.is_interrupted():
            logger.info(
                "Resuming from interrupted state at node '%s'", state.execution_meta.current_node
            )
            # This is a resume case - clear interrupt and merge input data
            if input_data:
                config["resume_data"] = input_data
                logger.debug("Added resume data with %d keys", len(input_data))
            state.clear_interrupt()
        elif not input_data.get("messages") and not state.context:
            # This is a fresh execution - validate input data
            error_msg = "Input data must contain 'messages' for new execution."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info(
                "Starting fresh execution with %d messages", len(input_data.get("messages", []))
            )

        return config

    async def _check_and_handle_interrupt(
        self,
        current_node: str,
        interrupt_type: str,
        state: AgentState,
        config: dict[str, Any],
    ) -> bool:
        """Check for interrupts and save state if needed. Returns True if interrupted."""
        interrupt_nodes = (
            self.interrupt_before if interrupt_type == "before" else self.interrupt_after
        )

        if current_node in interrupt_nodes:
            status = (
                ExecutionStatus.INTERRUPTED_BEFORE
                if interrupt_type == "before"
                else ExecutionStatus.INTERRUPTED_AFTER
            )
            state.set_interrupt(
                current_node,
                f"interrupt_{interrupt_type}: {current_node}",
                status,
            )
            # Save state and interrupt
            await sync_data(
                checkpointer=self.checkpointer,
                context_manager=self.state_graph.context_manager,
                state=state,
                config=config,
                messages=[],
                trim=True,
            )
            logger.debug("Node '%s' interrupted", current_node)
            return True

        logger.debug(
            "No interrupts found for node '%s', continuing execution",
            current_node,
        )
        return False

    async def _handle_sync_streaming(self, result: Any) -> AsyncIterator[StreamChunk]:
        """Handle synchronous streaming response."""
        logger.debug("Handling synchronous streaming response")
        for chunk in result:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta_content = ""

                if hasattr(choice, "delta") and choice.delta:
                    delta_content = getattr(choice.delta, "content", "") or ""

                finish_reason = getattr(choice, "finish_reason", None)

                yield StreamChunk(
                    delta=delta_content,
                    finish_reason=finish_reason,
                    is_final=finish_reason is not None,
                )

    async def _handle_async_streaming(self, result: Any) -> AsyncIterator[StreamChunk]:
        """Handle asynchronous streaming response."""
        logger.debug("Handling asynchronous streaming response")
        async for chunk in result:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta_content = ""
                # logger.debug("Processing async streaming chunk: %s", chunk)

                if hasattr(choice, "delta") and choice.delta:
                    delta_content = getattr(choice.delta, "content", "") or ""

                finish_reason = getattr(choice, "finish_reason", None)
                # logger.debug("Async streaming chunk finish reason: %s", finish_reason)

                yield StreamChunk(
                    delta=delta_content,
                    finish_reason=finish_reason,
                    is_final=finish_reason is not None,
                )

    async def _handle_non_streaming(
        self,
        result: Any,
        state: StateT,
    ) -> AsyncIterator[StreamChunk]:
        """Handle non-streaming response by simulating streaming."""
        # Extract content for streaming (don't process the result again here)
        logger.debug("Extracting content from non-streaming response")
        content = extract_content_from_response(result)
        logger.debug("Extracted content: %s", content)

        # Simulate streaming of the extracted content
        if content:
            logger.debug("Simulating streaming for content of length %d", len(content))
            async for chunk in simulate_async_streaming(content, delay=0.05):
                yield chunk
        else:
            # Empty response
            logger.debug("No content to stream, yielding empty final chunk")
            yield StreamChunk(content="", delta="", is_final=True, finish_reason="stop")

    async def _process_node_result_streaming(
        self,
        result: Any,
        state: StateT,
    ) -> AsyncIterator[StreamChunk]:
        """Process node result with streaming support."""
        # Check if result is a streaming response
        if is_streaming_response(result):
            logger.debug("Processing streaming response")
            async for chunk in self._handle_sync_streaming(result):
                yield chunk
        elif is_async_streaming_response(result):
            logger.debug("Processing async streaming response")
            async for chunk in self._handle_async_streaming(result):
                yield chunk
        else:
            logger.debug("Processing non-streaming response")
            async for chunk in self._handle_non_streaming(result, state):
                yield chunk

    async def _execute_graph_streaming(
        self,
        state: StateT,
        config: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Execute the entire graph with streaming support."""
        max_steps = config.get("recursion_limit", 25)

        # Get current execution info from state
        current_node = state.execution_meta.current_node
        step = state.execution_meta.step
        try:
            while current_node != END and step < max_steps:
                # Update execution metadata
                logger.debug(
                    "Executing node: %s (step: %d)",
                    current_node,
                    step,
                )
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(self.checkpointer, state, config)
                logger.debug("Realtime sync called")

                # Check for interrupt_before
                if await self._check_and_handle_interrupt(current_node, "before", state, config):
                    logger.info("Execution interrupted before node: %s", current_node)
                    yield StreamChunk(
                        content="", delta="", is_final=True, finish_reason="interrupted"
                    )
                    return

                ###############################################
                ##### Node Execution Started ##################
                ###############################################
                node = self.state_graph.nodes[current_node]
                result = await node.execute(
                    state,
                    config,
                    self.checkpointer,
                    self.store,
                    self.state_graph.dependency_container,
                    self.callback_manager,
                )
                ###############################################
                ##### Node Execution Done #####################
                ###############################################

                logger.debug("Node '%s' executed", current_node)

                # Process result using the regular logic to get proper next_node
                temp_messages: list[Message] = []

                try:
                    _, temp_messages, next_node = process_node_result(
                        result,
                        state,
                        temp_messages,
                    )

                    # If _process_node_result didn't return a next_node, use _get_next_node
                    if next_node is None and current_node:
                        next_node = get_next_node(current_node, state, self.state_graph.edges)

                except Exception:
                    # Log error silently and continue
                    logger.exception("Error processing node result at node '%s'", current_node)
                    next_node = None

                # For streaming, we yield chunks based on the result
                async for chunk in self._process_node_result_streaming(result, state):
                    yield chunk

                # Check for interrupt_after
                if await self._check_and_handle_interrupt(
                    current_node,
                    "after",
                    state,
                    config,
                ):
                    # For interrupt_after, advance to next node before pausing
                    if next_node is None and current_node:
                        next_node = get_next_node(current_node, state, self.state_graph.edges)
                    if next_node:
                        state.set_current_node(next_node)

                    yield StreamChunk(
                        content="", delta="", is_final=True, finish_reason="interrupted"
                    )
                    logger.info("Execution interrupted after node: %s", current_node)
                    return

                # Get next node (only if no explicit navigation from Command)
                if next_node is None:
                    current_node = get_next_node(current_node, state, self.state_graph.edges)
                else:
                    current_node = next_node

                # current node
                logger.debug(
                    "Current node: %s",
                    current_node,
                )

                # Advance step after successful node execution
                step += 1
                state.advance_step()
                await call_realtime_sync(self.checkpointer, state, config)
                logger.info(
                    "Graph execution progressed to step %d",
                    step,
                )

                if step >= max_steps:
                    state.error("Graph execution exceeded maximum steps")
                    logger.error("Graph execution exceeded maximum steps: %d", max_steps)
                    await call_realtime_sync(self.checkpointer, state, config)
                    raise GraphRecursionError(
                        f"Graph execution exceeded recursion limit: {max_steps}"
                    )

            # Execution completed successfully
            state.complete()
            await sync_data(
                checkpointer=self.checkpointer,
                context_manager=self.state_graph.context_manager,
                state=state,
                config=config,
                messages=[],
                trim=True,
            )
            logger.info("Graph execution completed successfully")

            # Yield final completion chunk
            yield StreamChunk(content="", delta="", is_final=True, finish_reason="stop")

        except Exception as e:
            # Handle execution errors
            logger.exception("Graph execution failed: %s", e)
            state.error(str(e))
            await sync_data(
                checkpointer=self.checkpointer,
                context_manager=self.state_graph.context_manager,
                state=state,
                config=config,
                messages=[],
                trim=True,
            )

            # Yield error chunk
            yield StreamChunk(content=str(e), delta=str(e), is_final=True, finish_reason="error")
            raise

    async def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the graph asynchronously with streaming support.

        Yields StreamChunk objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            StreamChunk objects with incremental content
        """
        config = config or {}
        input_data = input_data or {}

        # Load or initialize state
        new_state = await load_or_create_state(
            self.checkpointer,
            input_data,
            config,
            self.state_graph.state,
        )
        state: StateT = new_state  # type: ignore[assignment]
        logger.debug("Graph state loaded ")

        event = get_default_event(state, input_data=input_data, config=config)

        # Publish graph initialization event
        await self._publish_event(event)

        # Check if this is a resume case
        config = await self._check_interrupted(state, input_data, config)

        event.event_type = EventType.INVOKED
        await self._publish_event(event)

        # Execute graph with streaming
        async for chunk in self._execute_graph_streaming(state, config):
            yield chunk

        logger.info("Graph execution completed")
        event.event_type = EventType.COMPLETED
        await self._publish_event(event)

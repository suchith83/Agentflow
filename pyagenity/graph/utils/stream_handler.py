from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, AsyncIterable, TypeVar

from injectq import inject

from pyagenity.exceptions import GraphRecursionError
from pyagenity.graph.edge import Edge
from pyagenity.graph.node import Node
from pyagenity.publisher import BasePublisher, Event, EventType, SourceType
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.utils import (
    END,
    Message,
    ResponseGranularity,
    add_messages,
)
from pyagenity.utils.streaming import StreamChunk, StreamEvent

from .utils import (
    call_realtime_sync,
    get_default_event,
    get_next_node,
    load_or_create_state,
    process_node_result,
    sync_data,
)


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class StreamHandler[StateT: AgentState]:
    @inject
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        publisher: BasePublisher | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ):
        self.nodes: dict[str, Node] = nodes
        self.edges: list[Edge] = edges
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
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
        state: StateT,
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

    async def _execute_graph(
        self,
        state: StateT,
        config: dict[str, Any],
    ) -> AsyncIterable:
        """
        Execute the entire graph with support for interrupts and resuming.

        Why so many chunks are yielded?
        We allow user to set response type, if they want low granularity
        Only few chunks like Message will be sent to user
        """
        logger.info(
            "Starting graph execution from node '%s' at step %d",
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        messages: list[Message] = []
        max_steps = config.get("recursion_limit", 25)
        logger.debug("Max steps limit set to %d", max_steps)

        # Get current execution info from state
        current_node = state.execution_meta.current_node
        step = state.execution_meta.step

        event = get_default_event(
            state,
            input_data={},
            config=config,
            source=SourceType.NODE,
        )

        run_id = config.get("run_id", "")
        cfg = {
            "thread_id": config.get("thread_id", ""),
            "run_id": run_id,
            "run_timestamp": config.get("timestamp", ""),
        }

        chunk = StreamChunk(
            event=StreamEvent.NODE_EXECUTION,
            data={
                "state": state.model_dump(),
                "messages": [m.model_dump() for m in messages],
                "current_node": current_node,
                "step": step,
            },
            run_id=run_id,
            metadata=cfg,
        )

        run_id = config.get("run_id", "")
        cfg = {
            "thread_id": config.get("thread_id", ""),
            "run_id": run_id,
            "run_timestamp": config.get("timestamp", ""),
        }

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)
                # Lets update which node is being executed
                yield chunk

                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(state, config)
                event.payload["step"] = step
                event.payload["current_node"] = current_node
                event.event_type = EventType.RUNNING
                await self._publish_event(event)

                # Check for interrupt_before
                if await self._check_and_handle_interrupt(
                    current_node,
                    "before",
                    state,
                    config,
                ):
                    logger.info("Graph execution interrupted before node '%s'", current_node)
                    event.event_type = EventType.INTERRUPTED
                    event.payload["interrupted"] = "Before"
                    await self._publish_event(event)

                    # update chunk data before yielding
                    chunk.event = StreamEvent.INTERRUPTED
                    chunk.event_type = "Before"
                    chunk.data = {
                        "messages": [m.model_dump() for m in messages],
                    }

                    yield chunk
                    return

                # Execute current node
                logger.debug("Executing node '%s'", current_node)
                node = self.nodes[current_node]

                # Publish node invocation event
                await self._publish_event(event)

                ###############################################
                ##### Node Execution Started ##################
                ###############################################

                result = node.stream(config, state)  # type: ignore

                ###############################################
                ##### Node Execution Finished #################
                ###############################################

                logger.debug("Node '%s' execution completed", current_node)

                # Publish node completion event
                event.event_type = EventType.COMPLETED
                await self._publish_event(event)

                # Process result and get next node
                next_node = None
                async for rs in result:
                    if isinstance(rs, StreamChunk):
                        # nothing to block here for now
                        yield rs
                    elif isinstance(rs, dict) and "is_non_streaming" in rs:
                        state = rs.get("state", state)
                        messages = rs.get("messages", messages)
                        next_node = rs.get("next_node", next_node)
                    elif isinstance(rs, Message):
                        messages.append(rs)
                        logger.debug(
                            "Appended message from node '%s', total messages: %d",
                            current_node,
                            len(messages),
                        )
                    else:
                        # if nothing match try to process as node result
                        # if failed just log the error and continue
                        try:
                            state, messages, next_node = process_node_result(
                                rs,  # Pass the individual item, not the async generator
                                state,
                                messages,
                            )
                        except Exception as e:
                            logger.error("Failed to process node result: %s", e)

                logger.debug(
                    "Node result processed, next_node=%s, total_messages=%d",
                    next_node,
                    len(messages),
                )

                # Add collected messages to state context before determining next node
                if messages:
                    state.context = add_messages(state.context, messages)
                    logger.debug("Added %d messages to state context", len(messages))

                # Call realtime sync after node execution (if state/messages changed)
                await call_realtime_sync(state, config)

                # Check for interrupt_after
                if await self._check_and_handle_interrupt(
                    current_node,
                    "after",
                    state,
                    config,
                ):
                    logger.info("Graph execution interrupted after node '%s'", current_node)
                    # For interrupt_after, advance to next node before pausing
                    if next_node is None:
                        next_node = get_next_node(current_node, state, self.edges)
                    state.set_current_node(next_node)

                    event.event_type = EventType.INTERRUPTED
                    event.payload["interrupted"] = "After"
                    await self._publish_event(event)

                    yield StreamChunk(
                        event=StreamEvent.INTERRUPTED,
                        event_type="Before",
                        data={
                            "state": state.model_dump(),
                            "messages": [m.model_dump() for m in messages],
                        },
                        run_id=run_id,
                        metadata=cfg,
                    )
                    return

                # Get next node (only if no explicit navigation from Command)
                if next_node is None:
                    current_node = get_next_node(current_node, state, self.edges)
                    logger.debug("Next node determined by graph logic: '%s'", current_node)
                else:
                    current_node = next_node
                    logger.debug("Next node determined by command: '%s'", current_node)

                # Advance step after successful node execution
                step += 1
                state.advance_step()
                await call_realtime_sync(state, config)
                event.event_type = EventType.CHANGED

                event.payload["State_Updated"] = "State Updated"
                await self._publish_event(event)

                yield StreamChunk(
                    event=StreamEvent.STATE,
                    event_type="After",
                    data={
                        "state": state.model_dump(),
                        "step": step,
                    },
                    run_id=run_id,
                    metadata=cfg,
                )

                if step >= max_steps:
                    error_msg = "Graph execution exceeded maximum steps"
                    logger.error(error_msg)
                    state.error(error_msg)
                    await call_realtime_sync(state, config)
                    yield StreamChunk(
                        event=StreamEvent.ERROR,
                        event_type="Before",
                        data={"error": error_msg},
                        run_id=run_id,
                        metadata=cfg,
                    )

                    raise GraphRecursionError(
                        f"Graph execution exceeded recursion limit: {max_steps}"
                    )

            # Execution completed successfully
            logger.info(
                "Graph execution completed successfully at node '%s' after %d steps",
                current_node,
                step,
            )
            state.complete()
            is_context_trimmed = await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            yield StreamChunk(
                event=StreamEvent.COMPLETE,
                event_type="After",
                data={
                    "state": state.model_dump(),
                    "messages": [m.model_dump() for m in messages],
                    "context_trimmed": is_context_trimmed,
                },
                run_id=run_id,
                metadata=cfg,
            )

        except Exception as e:
            # Handle execution errors
            logger.exception("Graph execution failed: %s", e)

            # Publish error event
            event.event_type = EventType.ERROR
            event.payload["error"] = str(e)
            await self._publish_event(event)

            state.error(str(e))
            await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            raise

    async def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncGenerator[StreamChunk]:
        """Execute the graph asynchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state
        based on the AgentState's execution metadata.

        Args:
            input_data: Input dict with 'messages' key (for new execution) or
                       additional data for resuming
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Returns:
            Response dict based on granularity
        """
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        config = config or {}
        input_data = input_data or {}

        # Load or initialize state
        logger.debug("Loading or creating state from input data")
        new_state = await load_or_create_state(
            input_data,
            config,
            default_state,
        )
        state: StateT = new_state  # type: ignore[assignment]
        logger.debug(
            "State loaded: interrupted=%s, current_node=%s, step=%d",
            state.is_interrupted(),
            state.execution_meta.current_node,
            state.execution_meta.step,
        )

        event = get_default_event(state=state, input_data=input_data, config=config)

        # Publish graph initialization event
        await self._publish_event(event)

        # Check if this is a resume case
        config = await self._check_interrupted(state, input_data, config)

        # Publish graph error event
        event.event_type = EventType.INVOKED
        await self._publish_event(event)

        # Now start Execution
        # Execute graph
        logger.debug("Beginning graph execution")
        result = self._execute_graph(state, config)
        async for chunk in result:
            # only StreamChunk will be shared with caller
            # Other types are used for internal handling
            if isinstance(chunk, StreamChunk):
                yield chunk

        # Publish graph completion event
        event.event_type = EventType.COMPLETED
        await self._publish_event(event)

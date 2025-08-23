from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.checkpointer import BaseCheckpointer
from pyagenity.exceptions import GraphRecursionError
from pyagenity.publisher import BasePublisher, Event, EventType, SourceType
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.store import BaseStore
from pyagenity.utils import (
    END,
    CallbackManager,
    Message,
    ResponseGranularity,
    default_callback_manager,
)

from .utils import (
    call_realtime_sync,
    get_default_event,
    get_next_node,
    load_or_create_state,
    parse_response,
    process_node_result,
    sync_data,
)


# Import StateGraph only for typing to avoid circular import at runtime
if TYPE_CHECKING:
    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class InvokeHandler[StateT: AgentState]:
    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
        publisher: BasePublisher | None = None,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
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

    async def _execute_graph(
        self,
        state: StateT,
        config: dict[str, Any],
    ) -> tuple[StateT, list[Message]]:
        """Execute the entire graph with support for interrupts and resuming."""
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

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)
                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(self.checkpointer, state, config)
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
                    return state, messages

                # Execute current node
                logger.debug("Executing node '%s'", current_node)
                node = self.state_graph.nodes[current_node]

                # Publish node invocation event
                await self._publish_event(event)

                ###############################################
                ##### Node Execution Started ##################
                ###############################################

                result = await node.execute(
                    state,
                    config,
                    self.checkpointer,
                    self.store,
                    self.state_graph.dependency_container,
                    self.callback_manager,
                )

                ###############################################
                ##### Node Execution Finished #################
                ###############################################

                logger.debug("Node '%s' execution completed", current_node)

                # Publish node completion event
                event.event_type = EventType.COMPLETED
                await self._publish_event(event)

                # Process result and get next node
                state, messages, next_node = process_node_result(
                    result,
                    state,
                    messages,
                )
                logger.debug(
                    "Node result processed, next_node=%s, total_messages=%d",
                    next_node,
                    len(messages),
                )

                # Call realtime sync after node execution (if state/messages changed)
                await call_realtime_sync(self.checkpointer, state, config)

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
                        next_node = get_next_node(current_node, state, self.state_graph.edges)
                    state.set_current_node(next_node)

                    event.event_type = EventType.INTERRUPTED
                    event.payload["interrupted"] = "After"
                    await self._publish_event(event)
                    return state, messages

                # Get next node (only if no explicit navigation from Command)
                if next_node is None:
                    current_node = get_next_node(current_node, state, self.state_graph.edges)
                    logger.debug("Next node determined by graph logic: '%s'", current_node)
                else:
                    current_node = next_node
                    logger.debug("Next node determined by command: '%s'", current_node)

                # Advance step after successful node execution
                step += 1
                state.advance_step()
                await call_realtime_sync(self.checkpointer, state, config)
                event.event_type = EventType.CHANGED

                event.payload["State_Updated"] = "State Updated"
                await self._publish_event(event)

                if step >= max_steps:
                    error_msg = "Graph execution exceeded maximum steps"
                    logger.error(error_msg)
                    state.error(error_msg)
                    await call_realtime_sync(self.checkpointer, state, config)
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
            await sync_data(
                checkpointer=self.checkpointer,
                context_manager=self.state_graph.context_manager,
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            return state, messages

        except Exception as e:
            # Handle execution errors
            logger.exception("Graph execution failed: %s", e)

            # Publish error event
            event.event_type = EventType.ERROR
            event.payload["error"] = str(e)
            await self._publish_event(event)

            state.error(str(e))
            await sync_data(
                checkpointer=self.checkpointer,
                context_manager=self.state_graph.context_manager,
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            raise

    async def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
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
            self.checkpointer,
            input_data,
            config,
            self.state_graph.state,
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
        final_state, messages = await self._execute_graph(state, config)
        logger.info("Graph execution completed with %d final messages", len(messages))

        # Publish graph completion event
        event.event_type = EventType.COMPLETED
        await self._publish_event(event)

        return await parse_response(
            final_state,
            messages,
            response_granularity,
        )

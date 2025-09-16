from __future__ import annotations

import logging
from typing import Any, TypeVar

from injectq import inject

from pyagenity.exceptions import GraphRecursionError
from pyagenity.graph.edge import Edge
from pyagenity.graph.node import Node
from pyagenity.graph.utils.utils import (
    call_realtime_sync,
    get_next_node,
    load_or_create_state,
    parse_response,
    publish_event,
    sync_data,
)
from pyagenity.publisher import BasePublisher
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.utils import (
    END,
    Message,
    ResponseGranularity,
)
from pyagenity.utils.reducers import add_messages
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class InvokeHandler[StateT: AgentState]:
    @inject
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ):
        self.nodes: dict[str, Node] = nodes
        self.edges: list[Edge] = edges
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []

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

        # Create event for graph execution
        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=current_node,
            extra={
                "current_node": current_node,
                "step": step,
                "max_steps": max_steps,
            },
        )

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)
                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(state, config)
                event.data["state"] = state.model_dump()
                event.metadata["step"] = step
                event.metadata["current_node"] = current_node
                event.event_type = EventType.PROGRESS
                publish_event(event)

                # Check for interrupt_before
                if await self._check_and_handle_interrupt(
                    current_node,
                    "before",
                    state,
                    config,
                ):
                    logger.info("Graph execution interrupted before node '%s'", current_node)
                    event.event_type = EventType.INTERRUPTED
                    event.metadata["interrupted"] = "Before"
                    event.metadata["status"] = "Graph execution interrupted before node execution"
                    event.data["interrupted"] = "Before"
                    publish_event(event)
                    return state, messages

                # Execute current node
                logger.debug("Executing node '%s'", current_node)
                node = self.nodes[current_node]

                # Publish node invocation event

                ###############################################
                ##### Node Execution Started ##################
                ###############################################

                result = await node.execute(config, state)  # type: ignore

                ###############################################
                ##### Node Execution Finished #################
                ###############################################

                logger.debug("Node '%s' execution completed", current_node)

                next_node = None

                # Process result and get next node
                if isinstance(result, list):
                    # If result is a list of Message, append to messages
                    messages.extend(result)
                    logger.debug(
                        "Node '%s' returned %d messages, total messages now %d",
                        current_node,
                        len(result),
                        len(messages),
                    )
                    # Add messages to state context so they're visible to subsequent nodes
                    state.context = add_messages(state.context, result)

                # No state change beyond adding messages, just advance to next node
                if isinstance(result, dict):
                    state = result.get("state", state)
                    next_node = result.get("next_node")
                    new_messages = result.get("messages", [])
                    if new_messages:
                        messages.extend(new_messages)
                        logger.debug(
                            "Node '%s' returned %d messages, total messages now %d",
                            current_node,
                            len(new_messages),
                            len(messages),
                        )

                logger.debug(
                    "Node result processed, next_node=%s, total_messages=%d",
                    next_node,
                    len(messages),
                )

                # Call realtime sync after node execution (if state/messages changed)
                await call_realtime_sync(state, config)
                event.event_type = EventType.UPDATE
                event.data["state"] = state.model_dump()
                event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                event.content_type = [ContentType.STATE, ContentType.MESSAGE]
                publish_event(event)

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
                    event.data["interrupted"] = "After"
                    event.metadata["interrupted"] = "After"
                    event.data["state"] = state.model_dump()
                    publish_event(event)
                    return state, messages

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
                event.event_type = EventType.UPDATE

                event.metadata["State_Updated"] = "State Updated"
                event.data["state"] = state.model_dump()
                publish_event(event)

                if step >= max_steps:
                    error_msg = "Graph execution exceeded maximum steps"
                    logger.error(error_msg)
                    state.error(error_msg)
                    await call_realtime_sync(state, config)
                    event.event_type = EventType.ERROR
                    event.data["state"] = state.model_dump()
                    event.metadata["error"] = error_msg
                    event.metadata["step"] = step
                    event.metadata["current_node"] = current_node

                    publish_event(event)
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
            res = await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            event.event_type = EventType.END
            event.data["state"] = state.model_dump()
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            event.content_type = [ContentType.STATE, ContentType.MESSAGE]
            event.metadata["status"] = "Graph execution completed"
            event.metadata["step"] = step
            event.metadata["current_node"] = current_node
            event.metadata["is_context_trimmed"] = res

            publish_event(event)

            return state, messages

        except Exception as e:
            # Handle execution errors
            logger.exception("Graph execution failed: %s", e)
            state.error(str(e))

            # Publish error event
            event.event_type = EventType.ERROR
            event.metadata["error"] = str(e)
            event.data["state"] = state.model_dump()
            publish_event(event)

            await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            raise

    async def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
        """Execute the graph asynchronously with event publishing."""
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
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

        # Event publishing logic
        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=state.execution_meta.current_node,
            extra={
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
            },
        )
        event.event_type = EventType.START
        publish_event(event)

        # Check if this is a resume case
        config = await self._check_interrupted(state, input_data, config)

        event.event_type = EventType.UPDATE
        event.metadata["status"] = "Graph invoked"
        publish_event(event)

        try:
            logger.debug("Beginning graph execution")
            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Graph execution started"
            publish_event(event)

            final_state, messages = await self._execute_graph(state, config)
            logger.info("Graph execution completed with %d final messages", len(messages))

            event.event_type = EventType.END
            event.metadata["status"] = "Graph execution completed"
            event.data["state"] = final_state.model_dump()
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            publish_event(event)

            return await parse_response(
                final_state,
                messages,
                response_granularity,
            )
        except Exception as e:
            logger.exception("Graph execution failed: %s", e)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Graph execution failed: {e}"
            event.data["error"] = str(e)
            publish_event(event)
            raise

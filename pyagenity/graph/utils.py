from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from litellm.types.utils import ModelResponse

from pyagenity.checkpointer import BaseCheckpointer
from pyagenity.publisher import Event, EventType, SourceType
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.state.base_context import BaseContextManager
from pyagenity.state.execution_state import ExecutionState as ExecMeta
from pyagenity.utils import (
    END,
    START,
    Command,
    Message,
    ResponseGranularity,
    add_messages,
    call_sync_or_async,
)


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


async def parse_response(
    state: AgentState,
    messages: list[Message],
    response_granularity: ResponseGranularity = ResponseGranularity.LOW,
) -> dict[str, Any]:
    """Parse response based on granularity."""
    match response_granularity:
        case ResponseGranularity.FULL:
            # Return full state and messages
            return {"state": state, "messages": messages}
        case ResponseGranularity.PARTIAL:
            # Return state and summary of messages
            return {
                "state": None,
                "context": state.context,
                "summary": state.context_summary,
                "message": messages,
            }
        case ResponseGranularity.LOW:
            # Return only latest message
            return {"messages": messages}

    return {"messages": messages}


# Utility to update only provided fields in state
def _update_state_fields(state, partial: dict):
    """Update only the provided fields in the state object."""
    for k, v in partial.items():
        # Avoid updating special fields
        if k in ("context", "context_summary", "execution_meta"):
            continue
        if hasattr(state, k):
            setattr(state, k, v)


async def load_or_create_state[StateT: AgentState](
    checkpointer: BaseCheckpointer | None,
    input_data: dict[str, Any],
    config: dict[str, Any],
    old_state: StateT,
) -> StateT:
    """Load existing state from checkpointer or create new state.

    Attempts to fetch a realtime-synced state first, then falls back to
    the persistent checkpointer. If no existing state is found, creates
    a new state from the `StateGraph`'s prototype state and merges any
    incoming messages. Supports partial state update via 'state' in input_data.
    """
    logger.debug("Loading or creating state with thread_id=%s", config.get("thread_id", "default"))

    # Try to load existing state if checkpointer is available
    if checkpointer:
        logger.debug("Attempting to load existing state from checkpointer")
        # first check realtime-synced state
        existing_state: StateT = await call_sync_or_async(checkpointer.get_sync_state, config)
        if not existing_state:
            logger.debug("No synced state found, trying persistent checkpointer")
            # If no synced state, try to get from persistent checkpointer
            existing_state = await call_sync_or_async(checkpointer.get_state, config)

        if existing_state:
            logger.info(
                "Loaded existing state with %d context messages, current_node=%s, step=%d",
                len(existing_state.context) if existing_state.context else 0,
                existing_state.execution_meta.current_node,
                existing_state.execution_meta.step,
            )
            # Merge new messages with existing context
            new_messages = input_data.get("messages", [])
            if new_messages:
                logger.debug("Merging %d new messages with existing context", len(new_messages))
                existing_state.context = add_messages(existing_state.context, new_messages)
            # Merge partial state fields if provided
            partial_state = input_data.get("state", {})
            if partial_state and isinstance(partial_state, dict):
                logger.debug("Merging partial state with %d fields", len(partial_state))
                _update_state_fields(existing_state, partial_state)
            # Update current node if available
            if "current_node" in partial_state and partial_state["current_node"] is not None:
                existing_state.set_current_node(partial_state["current_node"])
            return existing_state
    else:
        logger.debug("No checkpointer available, will create new state")

    # Create new state by deep copying the graph's prototype state
    logger.info("Creating new state from graph prototype")
    import copy  # noqa: PLC0415

    state = copy.deepcopy(old_state)

    # Ensure core AgentState fields are properly initialized
    if hasattr(state, "context") and not isinstance(state.context, list):
        state.context = []
        logger.debug("Initialized empty context list")
    if hasattr(state, "context_summary") and state.context_summary is None:
        state.context_summary = None
        logger.debug("Initialized context_summary as None")
    if hasattr(state, "execution_meta"):
        # Create a fresh execution metadata
        state.execution_meta = ExecMeta(current_node=START)
        logger.debug("Created fresh execution metadata starting at %s", START)

    # Set thread_id in execution metadata
    thread_id = config.get("thread_id", "default")
    state.execution_meta.thread_id = thread_id
    logger.debug("Set thread_id to %s", thread_id)

    # Merge new messages with context
    new_messages = input_data.get("messages", [])
    if new_messages:
        logger.debug("Adding %d new messages to fresh state", len(new_messages))
        state.context = add_messages(state.context, new_messages)
    # Merge partial state fields if provided
    partial_state = input_data.get("state", {})
    if partial_state and isinstance(partial_state, dict):
        logger.debug("Merging partial state with %d fields", len(partial_state))
        _update_state_fields(state, partial_state)

    logger.info(
        "Created new state with %d context messages", len(state.context) if state.context else 0
    )
    if "current_node" in partial_state and partial_state["current_node"] is not None:
        state.set_current_node(partial_state["current_node"])
    return state  # type: ignore[return-value]


def get_default_event(
    state: AgentState,
    source: SourceType = SourceType.GRAPH,
    event_type: EventType = EventType.INITIALIZE,
    input_data: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> Event:
    metadata = meta or {}
    metadata["step"] = state.execution_meta.step
    metadata["current_node"] = state.execution_meta.current_node

    return Event(
        source=source,
        event_type=event_type,
        payload={
            "input_keys": list(input_data.keys()) if input_data else [],
            "is_resume": state.is_interrupted(),
        },
        meta=metadata,
        config=config or {},
    )


def process_node_result[StateT: AgentState](
    result: Any,
    state: StateT,
    messages: list[Message],
) -> tuple[StateT, list[Message], str | None]:
    """Process result from node execution and return updated state, messages, and next node."""
    next_node = None

    if isinstance(result, Command):
        # Apply state updates
        if result.update:
            if isinstance(result.update, ModelResponse):
                lm = Message.from_response(result.update)
                messages.append(lm)
                state.context = add_messages(state.context, [lm])
            elif isinstance(result.update, AgentState):
                # Check if it's an AgentState or subclass - cast to StateT
                state = result.update  # type: ignore[assignment]
                messages.append(
                    state.context[-1] if state.context else Message.from_text("Unknown")
                )
        # Handle navigation
        if result.goto:
            next_node = result.goto

    elif isinstance(result, Message):
        messages.append(result)
        state.context = add_messages(state.context, [result])

    elif isinstance(result, AgentState):
        # Check if result is an AgentState or subclass - cast to StateT
        state = result  # type: ignore[assignment]
        messages.append(state.context[-1] if state.context else Message.from_text("Unknown"))

    elif isinstance(result, dict):
        try:
            lm = Message.from_dict(result)
        except Exception as e:
            raise ValueError(f"Invalid message dict: {e}") from e
        messages.append(lm)
        state.context = add_messages(state.context, [lm])

    elif isinstance(result, str):
        lm = Message.from_text(result)
        messages.append(lm)
        state.context = add_messages(state.context, [lm])

    elif isinstance(result, ModelResponse):
        lm = Message.from_response(result)
        messages.append(lm)
        state.context = add_messages(state.context, [lm])

    return state, messages, next_node


async def check_and_handle_interrupt(
    interrupt_before: list[str],
    interrupt_after: list[str],
    current_node: str,
    interrupt_type: str,
    state: AgentState,
    config: dict[str, Any],
    _sync_data: Callable,
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    interrupt_nodes = interrupt_before if interrupt_type == "before" else interrupt_after

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
        await _sync_data(state, config, [])
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


def get_next_node(
    current_node: str,
    state: AgentState,
    edges: list,
) -> str:
    """Get the next node to execute based on edges."""
    # Find outgoing edges from current node
    outgoing_edges = [e for e in edges if e.from_node == current_node]

    if not outgoing_edges:
        logger.debug("No outgoing edges from node '%s', ending execution", current_node)
        return END

    # Handle conditional edges
    for edge in outgoing_edges:
        if edge.condition:
            try:
                condition_result = edge.condition(state)
                if hasattr(edge, "condition_result"):
                    # Mapped conditional edge
                    if condition_result == edge.condition_result:
                        return edge.to_node
                elif isinstance(condition_result, str):
                    return condition_result
                elif condition_result:
                    return edge.to_node
            except Exception:
                logger.exception("Error evaluating condition for edge: %s", edge)
                continue

    # Return first static edge if no conditions matched
    static_edges = [e for e in outgoing_edges if not e.condition]
    if static_edges:
        return static_edges[0].to_node

    logger.debug("No valid edges found from node '%s', ending execution", current_node)
    return END


async def call_realtime_sync(
    checkpointer: BaseCheckpointer | None,
    state: AgentState,
    config: dict[str, Any],
) -> None:
    """Call the realtime state sync hook if provided."""
    if checkpointer:
        logger.debug("Calling realtime state sync hook")
        await call_sync_or_async(checkpointer.sync_state, config, state)


async def sync_data(
    checkpointer: BaseCheckpointer | None,
    context_manager: BaseContextManager | None,
    state: AgentState,
    config: dict[str, Any],
    messages: list[Message],
    trim: bool = False,
) -> None:
    """Sync the current state and messages to the checkpointer."""
    if not checkpointer:
        logger.debug("No checkpointer available, skipping sync")
        return  # Nothing to do

    import copy  # noqa: PLC0415

    new_state = copy.deepcopy(state)
    # if context manager is available then utilize it
    if context_manager and trim:
        new_state = await call_sync_or_async(
            context_manager.trim_context,
            state,
        )

    # first sync with realtime then main db
    await call_sync_or_async(checkpointer.sync_state, config, state)
    logger.debug("Persisting state and %d messages to checkpointer", len(messages))

    await call_sync_or_async(checkpointer.put_state, config, new_state)
    if messages:
        await call_sync_or_async(checkpointer.put_messages, config, messages)

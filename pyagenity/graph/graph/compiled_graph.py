import asyncio
from collections.abc import Callable
from tokenize import maybe
from typing import TYPE_CHECKING, Any

from litellm.types.utils import ModelResponse

from pyagenity.graph.checkpointer import BaseCheckpointer, BaseStore
from pyagenity.graph.exceptions import GraphRecursionError
from pyagenity.graph.state import AgentState
from pyagenity.graph.state.execution_state import ExecutionStatus
from pyagenity.graph.utils import (
    END,
    Command,
    Message,
    ResponseGranularity,
    add_messages,
)
from pyagenity.graph.utils.callable_utils import call_sync_or_async


if TYPE_CHECKING:
    from .state_graph import StateGraph


class CompiledGraph:
    """A compiled graph ready for execution."""

    def __init__(
        self,
        state_graph: "StateGraph",
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.debug = debug
        self.context_manager = state_graph.context_manager
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph synchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state.

        Args:
            input_data: Input dict (no longer accepts AgentState directly)
            config: Configuration dictionary

        Returns:
            Final state dict and messages
        """
        return asyncio.run(self.ainvoke(input_data, config, response_granularity))

    async def ainvoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
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
        config = config or {}
        input_data = input_data or {}

        # Load or initialize state
        state = await self._load_or_create_state(input_data, config)

        # Check if this is a resume case
        if state.is_interrupted():
            # This is a resume case - clear interrupt and merge input data
            if input_data:
                config["resume_data"] = input_data
            state.clear_interrupt()
        elif not input_data.get("messages") and not state.context:
            # This is a fresh execution - validate input data
            raise ValueError("Input data must contain 'messages' for new execution.")

        # Execute graph
        final_state, messages = await self._execute_graph(state, config)

        return await self.parse_response(
            final_state,
            messages,
            response_granularity,
        )

    async def parse_response(
        self,
        state: AgentState,
        messages: list[Message],
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Parse response based on granularity."""
        match response_granularity:
            case ResponseGranularity.FULL:
                # Return full state and messages
                return {"state": state.to_dict(include_internal=False), "messages": messages}
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

    async def _load_or_create_state(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> AgentState:
        """Load existing state from checkpointer or create new state."""
        # Try to load existing state if checkpointer is available
        if self.checkpointer:
            # now first check we have state in realtime sync options
            existing_state = self.checkpointer.get_sync_state(config)
            if not existing_state:
                # If no synced state, try to get from checkpointer
                existing_state = self.checkpointer.get_state(config)

            if existing_state:
                # Merge new messages with existing context
                new_messages = input_data.get("messages", [])
                if new_messages:
                    existing_state.context = add_messages(existing_state.context, new_messages)
                return existing_state

        # Create new state
        state: AgentState = self.state_graph.state
        # Set thread_id in execution metadata
        thread_id = config.get("thread_id", "default")
        state.execution_meta.thread_id = thread_id

        # Merge new messages with context
        new_messages = input_data.get("messages", [])
        if new_messages:
            state.context = add_messages(state.context, new_messages)
        return state

    async def _call_realtime_sync(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> None:
        """Call the realtime state sync hook if provided."""
        if self.checkpointer:
            await call_sync_or_async(self.checkpointer.sync_state, config, state)

    async def sync_data(
        self,
        state: AgentState,
        config: dict[str, Any],
        messages: list[Message],
        trim: bool = False,
    ) -> None:
        """Sync the current state and messages to the checkpointer."""
        if not self.checkpointer:
            return  # Nothing to do

        new_state = state
        # if context manager is available then utilize it
        if self.context_manager and trim:
            new_state = await call_sync_or_async(
                self.context_manager.trim_context,
                state,
            )

        # first sync with realtime then main db
        await self._call_realtime_sync(new_state, config)

        await call_sync_or_async(self.checkpointer.put_state, config, new_state)
        if messages:
            await call_sync_or_async(self.checkpointer.put_messages, config, messages)

    async def _execute_graph(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> tuple[AgentState, list[Message]]:
        """Execute the entire graph with support for interrupts and resuming."""
        messages: list[Message] = []
        max_steps = config.get("recursion_limit", 25)

        # Get current execution info from state
        current_node = state.execution_meta.current_node
        step = state.execution_meta.step

        try:
            while current_node != END and step < max_steps:
                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await self._call_realtime_sync(state, config)

                # Check for interrupt_before
                if await self._check_and_handle_interrupt(current_node, "before", state, config):
                    return state, messages

                # Execute current node
                node = self.state_graph.nodes[current_node]
                result = await node.execute(
                    state,
                    config,
                    self.checkpointer,
                    self.store,
                )

                # Process result and get next node
                state, messages, next_node = await self._process_node_result(
                    result,
                    state,
                    messages,
                )

                # Call realtime sync after node execution (if state/messages changed)
                await self._call_realtime_sync(state, config)

                # Check for interrupt_after
                if await self._check_and_handle_interrupt(
                    current_node,
                    "after",
                    state,
                    config,
                ):
                    # For interrupt_after, advance to next node before pausing
                    if next_node is None:
                        next_node = self._get_next_node(current_node, state)
                    state.set_current_node(next_node)

                    return state, messages

                # Get next node (only if no explicit navigation from Command)
                if next_node is None:
                    current_node = self._get_next_node(current_node, state)
                else:
                    current_node = next_node

                # Advance step after successful node execution
                step += 1
                state.advance_step()
                await self._call_realtime_sync(state, config)

                if step >= max_steps:
                    state.error("Graph execution exceeded maximum steps")
                    await self._call_realtime_sync(state, config)
                    raise GraphRecursionError(
                        f"Graph execution exceeded recursion limit: {max_steps}"
                    )

            # Execution completed successfully
            state.complete()
            await self.sync_data(state, config, messages, trim=True)
            return state, messages

        except Exception as e:
            # Handle execution errors
            state.error(str(e))
            await self.sync_data(state, config, messages, trim=True)
            raise

    async def _process_node_result(
        self, result: Any, state: AgentState, messages: list[Message]
    ) -> tuple[AgentState, list[Message], str | None]:
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
                    state = result.update
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
            state = result
            messages.append(state.context[-1] if state.context else Message.from_text("Unknown"))

        elif isinstance(result, dict):
            lm = Message.from_dict(result)
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
            await self.sync_data(state, config, [])
            return True
        return False

    def _get_next_node(
        self,
        current_node: str,
        state: AgentState,
    ) -> str:
        """Get the next node to execute based on edges."""
        # Find outgoing edges from current node
        outgoing_edges = [e for e in self.state_graph.edges if e.from_node == current_node]

        if not outgoing_edges:
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
                    if self.debug:
                        # Debug logging could be added here if needed
                        pass
                    continue

        # Return first static edge if no conditions matched
        static_edges = [e for e in outgoing_edges if not e.condition]
        if static_edges:
            return static_edges[0].to_node

        return END

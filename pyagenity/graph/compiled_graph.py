from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from typing import TYPE_CHECKING, Any, TypeVar

from litellm.types.utils import ModelResponse

from pyagenity.checkpointer import BaseCheckpointer
from pyagenity.exceptions import GraphRecursionError
from pyagenity.state import AgentState, ExecutionStatus
from pyagenity.state.execution_state import ExecutionState as ExecMeta
from pyagenity.store import BaseStore
from pyagenity.utils import (
    END,
    START,
    CallbackManager,
    Command,
    Message,
    ResponseGranularity,
    StreamChunk,
    add_messages,
    call_sync_or_async,
    default_callback_manager,
    extract_content_from_response,
    is_async_streaming_response,
    is_streaming_response,
    simulate_async_streaming,
)


# Import StateGraph only for typing to avoid circular import at runtime
if TYPE_CHECKING:
    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)


# Utility to update only provided fields in state
def _update_state_fields(state, partial: dict):
    """Update only the provided fields in the state object."""
    for k, v in partial.items():
        # Avoid updating special fields
        if k in ("context", "context_summary", "execution_meta"):
            continue
        if hasattr(state, k):
            setattr(state, k, v)


class CompiledGraph[StateT: AgentState]:
    """A compiled graph ready for execution.

    Generic over state types to support custom AgentState subclasses.
    """

    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.debug = debug
        self.context_manager = state_graph.context_manager
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self.callback_manager = callback_manager or default_callback_manager

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph synchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state.

        Args:
            input_data: Input dict
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

        return await self._parse_response(
            final_state,
            messages,
            response_granularity,
        )

    async def _parse_response(
        self,
        state: StateT,
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

    def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> Generator[StreamChunk, None, None]:
        """Execute the graph synchronously with streaming support.

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

        # For sync streaming, we'll use asyncio.run to handle the async implementation
        async def _async_stream():
            async for chunk in self.astream(input_data, config, response_granularity):
                yield chunk

        # Use a helper to convert async generator to sync generator
        gen = _async_stream()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    async def astream(
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

        # Execute graph with streaming
        async for chunk in self._execute_graph_streaming(state, config):
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
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await self._call_realtime_sync(state, config)

                # Check for interrupt_before
                if await self._check_and_handle_interrupt(current_node, "before", state, config):
                    yield StreamChunk(
                        content="", delta="", is_final=True, finish_reason="interrupted"
                    )
                    return

                # Execute current node

                node = self.state_graph.nodes[current_node]
                result = await node.execute(
                    state,
                    config,
                    self.checkpointer,
                    self.store,
                    self.state_graph.dependency_container,
                    self.callback_manager,
                )

                # Process result using the regular logic to get proper next_node
                temp_messages: list[Message] = []

                try:
                    _, temp_messages, next_node = await self._process_node_result(
                        result,
                        state,
                        temp_messages,
                    )

                    # If _process_node_result didn't return a next_node, use _get_next_node
                    if next_node is None and current_node:
                        next_node = self._get_next_node(current_node, state)

                except Exception:
                    # Log error silently and continue
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
                        next_node = self._get_next_node(current_node, state)
                    if next_node:
                        state.set_current_node(next_node)

                    yield StreamChunk(
                        content="", delta="", is_final=True, finish_reason="interrupted"
                    )
                    return

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
            await self._sync_data(state, config, [], trim=True)

            # Yield final completion chunk
            yield StreamChunk(content="", delta="", is_final=True, finish_reason="stop")

        except Exception as e:
            # Handle execution errors
            state.error(str(e))
            await self._sync_data(state, config, [], trim=True)

            # Yield error chunk
            yield StreamChunk(content=str(e), delta=str(e), is_final=True, finish_reason="error")
            raise

    async def _process_node_result_streaming(
        self,
        result: Any,
        state: StateT,
    ) -> AsyncIterator[StreamChunk]:
        """Process node result with streaming support."""
        # Check if result is a streaming response
        if is_streaming_response(result):
            async for chunk in self._handle_sync_streaming(result):
                yield chunk
        elif is_async_streaming_response(result):
            async for chunk in self._handle_async_streaming(result):
                yield chunk
        else:
            async for chunk in self._handle_non_streaming(result, state):
                yield chunk

    async def _handle_sync_streaming(self, result: Any) -> AsyncIterator[StreamChunk]:
        """Handle synchronous streaming response."""
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
        async for chunk in result:
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

    async def _handle_non_streaming(
        self,
        result: Any,
        state: StateT,
    ) -> AsyncIterator[StreamChunk]:
        """Handle non-streaming response by simulating streaming."""
        # Extract content for streaming (don't process the result again here)
        content = extract_content_from_response(result)

        # Simulate streaming of the extracted content
        if content:
            async for chunk in simulate_async_streaming(content, delay=0.05):
                yield chunk
        else:
            # Empty response
            yield StreamChunk(content="", delta="", is_final=True, finish_reason="stop")

    async def _load_or_create_state(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> StateT:
        """Load existing state from checkpointer or create new state.

        Attempts to fetch a realtime-synced state first, then falls back to
        the persistent checkpointer. If no existing state is found, creates
        a new state from the `StateGraph`'s prototype state and merges any
        incoming messages. Supports partial state update via 'state' in input_data.
        """
        # Try to load existing state if checkpointer is available
        if self.checkpointer:
            # first check realtime-synced state
            existing_state: StateT = await call_sync_or_async(
                self.checkpointer.get_sync_state, config
            )
            if not existing_state:
                # If no synced state, try to get from persistent checkpointer
                existing_state = await call_sync_or_async(self.checkpointer.get_state, config)

            if existing_state:
                # Merge new messages with existing context
                new_messages = input_data.get("messages", [])
                if new_messages:
                    existing_state.context = add_messages(existing_state.context, new_messages)
                # Merge partial state fields if provided
                partial_state = input_data.get("state")
                if partial_state and isinstance(partial_state, dict):
                    _update_state_fields(existing_state, partial_state)
                return existing_state

        # Create new state by deep copying the graph's prototype state
        import copy  # noqa: PLC0415

        state = copy.deepcopy(self.state_graph.state)

        # Ensure core AgentState fields are properly initialized
        if hasattr(state, "context") and not isinstance(state.context, list):
            state.context = []
        if hasattr(state, "context_summary") and state.context_summary is None:
            state.context_summary = None
        if hasattr(state, "execution_meta"):
            # Create a fresh execution metadata
            state.execution_meta = ExecMeta(current_node=START)

        # Set thread_id in execution metadata
        thread_id = config.get("thread_id", "default")
        state.execution_meta.thread_id = thread_id

        # Merge new messages with context
        new_messages = input_data.get("messages", [])
        if new_messages:
            state.context = add_messages(state.context, new_messages)
        # Merge partial state fields if provided
        partial_state = input_data.get("state")
        if partial_state and isinstance(partial_state, dict):
            _update_state_fields(state, partial_state)
        return state  # type: ignore[return-value]

    async def _call_realtime_sync(
        self,
        state: StateT,
        config: dict[str, Any],
    ) -> None:
        """Call the realtime state sync hook if provided."""
        if self.checkpointer:
            await call_sync_or_async(self.checkpointer.sync_state, config, state)

    async def _sync_data(
        self,
        state: StateT,
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
        state: StateT,
        config: dict[str, Any],
    ) -> tuple[StateT, list[Message]]:
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
                    self.state_graph.dependency_container,
                    self.callback_manager,
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
            await self._sync_data(state, config, messages, trim=True)
            return state, messages

        except Exception as e:
            # Handle execution errors
            state.error(str(e))
            await self._sync_data(state, config, messages, trim=True)
            raise

    async def _process_node_result(
        self, result: Any, state: StateT, messages: list[Message]
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
            await self._sync_data(state, config, [])
            return True
        return False

    def _get_next_node(
        self,
        current_node: str,
        state: StateT,
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

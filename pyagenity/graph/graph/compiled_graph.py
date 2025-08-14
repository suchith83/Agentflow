from typing import Any, Dict, Optional
import asyncio
from litellm.types.utils import ModelResponse
from pyagenity.graph.checkpointer import BaseCheckpointer
from pyagenity.graph.checkpointer import BaseStore
from pyagenity.graph.exceptions import GraphRecursionError
from .state_graph import StateGraph
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import (
    Command,
    END,
    START,
    ResponseGranularity,
    Message,
    add_messages,
)


class CompiledGraph:
    """A compiled graph ready for execution."""

    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: Optional[BaseCheckpointer] = None,
        store: Optional[BaseStore] = None,
        debug: bool = False,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.debug = debug
        self.context_manager = state_graph.context_manager

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the graph synchronously."""
        return asyncio.run(self.ainvoke(input_data, config))

    # def stream(
    #     self,
    #     input_data: Optional[Dict[str, Any]] = None,
    #     config: Optional[Dict[str, Any]] = None,
    # ):
    #     """Stream graph execution synchronously."""

    #     async def _async_stream():
    #         async for result in self.astream(input_data, config):
    #             yield result

    #     # Create a new event loop to run the async generator
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:

    #         async def run_generator():
    #             results = []
    #             async for result in self.astream(input_data, config):
    #                 results.append(result)
    #             return results

    #         results = loop.run_until_complete(run_generator())
    #         for result in results:
    #             yield result
    #     finally:
    #         loop.close()

    async def parse_response(
        self,
        state: AgentState,
        messages: list[Message],
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
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

    async def ainvoke(
        self,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> Dict[str, Any]:
        """Execute the graph asynchronously."""
        input_data = input_data or {}
        config = config or {}

        # Validate input data
        if not input_data.get("messages"):
            raise ValueError("Input data must contain 'messages'.")

        # Initialize state
        state = self._initialize_state(input_data, config)

        # Execute graph
        final_state, messages = await self._execute_graph(
            state,
            config,
        )

        return await self.parse_response(
            final_state,
            messages,
            response_granularity,
        )

    # async def astream(
    #     self,
    #     input_data: Optional[Dict[str, Any]] = None,
    #     config: Optional[Dict[str, Any]] = None,
    # ):
    #     """Stream graph execution asynchronously."""
    #     input_data = input_data or {}
    #     config = config or {}

    #     # Initialize state
    #     state = self._initialize_state(input_data, config)

    #     # Stream execution
    #     async for step_result in self._stream_graph(state, config):
    #         yield step_result

    def _initialize_state(
        self,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> AgentState:
        """Initialize the graph state."""
        state: AgentState = self.state_graph.state
        # Merge new messages with existing context
        new_messages = input_data.get("messages", [])
        if new_messages:
            state.context = add_messages(state.context, new_messages)
        return state

    async def _execute_graph(
        self,
        state: AgentState,
        config: Dict[str, Any],
    ) -> tuple[AgentState, list[Message]]:
        """Execute the entire graph."""
        current_node = START
        max_steps = config.get("recursion_limit", 25)
        step = 0
        messages: list[Message] = []

        while current_node != END and step < max_steps:
            # Execute current node
            node = self.state_graph.nodes[current_node]

            if self.debug:
                print(f"Executing node: {current_node}")

            result = await node.execute(
                state,
                config,
                self.checkpointer,
                self.store,
            )

            # Handle Command returns
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
                            state.context[-1]
                            if state.context
                            else Message.from_text("Unknown")
                        )
                # Handle navigation
                if result.goto:
                    current_node = result.goto

            elif isinstance(result, Message):
                messages.append(result)
                # update the start also
                state.context = add_messages(state.context, [result])

            elif isinstance(result, AgentState):
                state = result
                messages.append(
                    state.context[-1] if state.context else Message.from_text("Unknown")
                )

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

            else:
                print("Nothing returned from node execution")

            current_node = self._get_next_node(current_node, state)

            step += 1
            state.step = step

            if step >= max_steps:
                raise GraphRecursionError(
                    f"Graph execution exceeded recursion limit: {max_steps}"
                )

        return state, messages

    # async def _stream_graph(
    #     self,
    #     state: AgentState,
    #     config: Dict[str, Any],
    # ):
    #     """Stream graph execution step by step."""
    #     current_node = START
    #     max_steps = config.get("recursion_limit", 100)
    #     step = 0

    #     while current_node != END and step < max_steps:
    #         # Execute current node
    #         node: Node = self.state_graph.nodes[current_node]

    #         result = await node.execute(state, config)

    #         # Handle Command returns
    #         if isinstance(result, Command):
    #             # Apply state updates
    #             if result.update:
    #                 state.update(result.update)
    #                 # Fixme: Update context if messages changed
    #                 # state = self.context_manager.update_context(state)

    #             # Yield step result
    #             yield {current_node: result.update or {}}

    #             # Handle navigation
    #             if result.goto:
    #                 current_node = result.goto
    #             else:
    #                 current_node = self._get_next_node(current_node, state)
    #         else:
    #             # Apply state updates from dict return
    #             if isinstance(result, dict):
    #                 state.update(result)
    #                 # Fixme: later
    #                 # state = self.context_manager.update_context(state)

    #             # Yield step result
    #             yield {current_node: result}

    #             # Get next node via edges
    #             current_node = self._get_next_node(current_node, state)

    #         step += 1
    #         if "step" in state:
    #             state["step"] = step

    #     if step >= max_steps:
    #         raise GraphRecursionError(
    #             f"Graph execution exceeded recursion limit: {max_steps}"
    #         )

    def _get_next_node(
        self,
        current_node: str,
        state: AgentState,
    ) -> str:
        """Get the next node to execute based on edges."""
        # Find outgoing edges from current node
        outgoing_edges = [
            e for e in self.state_graph.edges if e.from_node == current_node
        ]

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
                    else:
                        # Direct conditional edge
                        if isinstance(condition_result, str):
                            return condition_result
                        elif condition_result:
                            return edge.to_node
                except Exception as e:
                    if self.debug:
                        print(f"Error in condition: {e}")
                    continue

        # Return first static edge if no conditions matched
        static_edges = [e for e in outgoing_edges if not e.condition]
        if static_edges:
            return static_edges[0].to_node

        return END

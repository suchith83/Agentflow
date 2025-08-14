from typing import Any, Dict, Optional
from pyagenity.graph.exceptions.recursion_error import GraphRecursionError
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.state import AgentState
from pyagenity.graph.utils.command import Command
from pyagenity.graph.utils.constants import END, START
from pyagenity.graph.utils.reducers import add_messages


class CompiledGraph:
    """A compiled graph ready for execution."""

    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: Optional[Any] = None,
        store: Optional[Any] = None,
        debug: bool = False,
    ):
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.debug = debug
        self.context_manager = state_graph.context_manager

    def invoke(
        self,
        input_data: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute the graph synchronously."""
        return asyncio.run(self.ainvoke(input_data, config))

    def stream(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Stream graph execution synchronously."""

        async def _async_stream():
            async for result in self.astream(input_data, config):
                yield result

        # Create a new event loop to run the async generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:

            async def run_generator():
                results = []
                async for result in self.astream(input_data, config):
                    results.append(result)
                return results

            results = loop.run_until_complete(run_generator())
            for result in results:
                yield result
        finally:
            loop.close()

    async def ainvoke(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the graph asynchronously."""
        input_data = input_data or {}
        config = config or {}

        # Initialize state
        state = self._initialize_state(input_data, config)

        # Execute graph
        final_state = await self._execute_graph(state, config)

        return final_state

    async def astream(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Stream graph execution asynchronously."""
        input_data = input_data or {}
        config = config or {}

        # Initialize state
        state = self._initialize_state(input_data, config)

        # Stream execution
        async for step_result in self._stream_graph(state, config):
            yield step_result

    def _initialize_state(
        self, input_data: Dict[str, Any], config: Dict[str, Any]
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
    ) -> Dict[str, Any]:
        """Execute the entire graph."""
        current_node = START
        max_steps = config.get("recursion_limit", 25)
        step = 0

        while current_node != END and step < max_steps:
            # Execute current node
            node = self.state_graph.nodes[current_node]

            if self.debug:
                print(f"Executing node: {current_node}")

            result = await node.execute(state, config)

            # Handle Command returns
            if isinstance(result, Command):
                # Apply state updates
                if result.update:
                    state.update(result.update)
                    # FIXME: Update context if messages changed
                    # if "messages" in result.update:
                    #     state = self.context_manager.update_context(state)

                # Handle navigation
                if result.goto:
                    current_node = result.goto
                else:
                    current_node = self._get_next_node(current_node, state)
            else:
                # Apply state updates from dict return
                if isinstance(result, dict):
                    state.update(result)
                    # Fixme: Update context if messages changed
                    # if "messages" in result:
                    #     state = self.context_manager.update_context(state)

                # Get next node via edges
                current_node = self._get_next_node(current_node, state)

            step += 1
            if "step" in state:
                state["step"] = step

        if step >= max_steps:
            raise GraphRecursionError(
                f"Graph execution exceeded recursion limit: {max_steps}"
            )

        return state

    async def _stream_graph(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """Stream graph execution step by step."""
        current_node = START
        max_steps = config.get("recursion_limit", 100)
        step = 0

        while current_node != END and step < max_steps:
            # Execute current node
            node = self.state_graph.nodes[current_node]

            result = await node.execute(state, config)

            # Handle Command returns
            if isinstance(result, Command):
                # Apply state updates
                if result.update:
                    state.update(result.update)
                    # Fixme: Update context if messages changed
                    # state = self.context_manager.update_context(state)

                # Yield step result
                yield {current_node: result.update or {}}

                # Handle navigation
                if result.goto:
                    current_node = result.goto
                else:
                    current_node = self._get_next_node(current_node, state)
            else:
                # Apply state updates from dict return
                if isinstance(result, dict):
                    state.update(result)
                    # Fixme: later
                    # state = self.context_manager.update_context(state)

                # Yield step result
                yield {current_node: result}

                # Get next node via edges
                current_node = self._get_next_node(current_node, state)

            step += 1
            if "step" in state:
                state["step"] = step

        if step >= max_steps:
            raise GraphRecursionError(
                f"Graph execution exceeded recursion limit: {max_steps}"
            )

    def _get_next_node(self, current_node: str, state: Dict[str, Any]) -> str:
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

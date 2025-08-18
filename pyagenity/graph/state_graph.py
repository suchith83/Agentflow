from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, Union

from pyagenity.checkpointer import BaseCheckpointer, BaseStore
from pyagenity.exceptions import GraphError
from pyagenity.state import AgentState, BaseContextManager
from pyagenity.utils import END, START, DependencyContainer

from .edge import Edge
from .node import Node
from .tool_node import ToolNode


if TYPE_CHECKING:
    from .compiled_graph import CompiledGraph


# Generic type variable bound to AgentState for state subtyping
StateT = TypeVar("StateT", bound=AgentState)


class StateGraph[StateT: AgentState]:
    """Main graph class for orchestrating multi-agent workflows.

    This class provides the core functionality for building and managing stateful
    agent workflows. It is similar to LangGraph's StateGraph but designed for
    direct LiteLLM integration with support for dependency injection.

    The graph is generic over state types to support custom AgentState subclasses,
    allowing for type-safe state management throughout the workflow execution.

    Attributes:
        state (StateT): The current state of the graph workflow.
        nodes (dict[str, Node]): Collection of nodes in the graph.
        edges (list[Edge]): Collection of edges connecting nodes.
        entry_point (str | None): Name of the starting node for execution.
        context_manager (BaseContextManager[StateT] | None): Optional context manager
            for handling cross-node state operations.
        dependency_container (DependencyContainer): Container for managing
            dependencies that can be injected into node functions.
        compiled (bool): Whether the graph has been compiled for execution.

    Example:
        >>> graph = StateGraph()
        >>> graph.add_node("process", process_function)
        >>> graph.add_edge(START, "process")
        >>> graph.add_edge("process", END)
        >>> compiled = graph.compile()
        >>> result = compiled.invoke({"input": "data"})
    """

    def __init__(
        self,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        dependency_container: DependencyContainer | None = None,
    ):
        """Initialize a new StateGraph instance.

        Args:
            state: Initial state for the graph. If None, a default AgentState
                will be created.
            context_manager: Optional context manager for handling cross-node
                state operations and advanced state management patterns.
            dependency_container: Container for managing dependencies that can
                be injected into node functions. If None, a new empty container
                will be created.

        Note:
            START and END nodes are automatically added to the graph upon
            initialization and accept the full node signature including
            dependencies.
        """
        # Initialize state and structure
        self.state = state or AgentState()  # type: ignore[assignment]
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: str | None = None
        self.context_manager: BaseContextManager[StateT] | None = context_manager
        self.dependency_container = dependency_container or DependencyContainer()
        self.compiled = False

        # Add START and END nodes (accept full node signature including dependencies)
        self.nodes[START] = Node(START, lambda state, config, **deps: state)
        self.nodes[END] = Node(END, lambda state, config, **deps: state)

    def add_node(
        self,
        name_or_func: str | Callable,
        func: Union[Callable, "ToolNode", None] = None,
    ) -> "StateGraph":
        """Add a node to the graph.

        This method supports two calling patterns:
        1. Pass a callable as the first argument (name inferred from function name)
        2. Pass a name string and callable/ToolNode as separate arguments

        Args:
            name_or_func: Either the node name (str) or a callable function.
                If callable, the function name will be used as the node name.
            func: The function or ToolNode to execute. Required if name_or_func
                is a string, ignored if name_or_func is callable.

        Returns:
            StateGraph: The graph instance for method chaining.

        Raises:
            ValueError: If invalid arguments are provided.

        Example:
            >>> # Method 1: Function name inferred
            >>> graph.add_node(my_function)
            >>> # Method 2: Explicit name and function
            >>> graph.add_node("process", my_function)
        """
        if callable(name_or_func) and func is None:
            # Function passed as first argument
            name = name_or_func.__name__
            func = name_or_func
        elif isinstance(name_or_func, str) and (callable(func) or isinstance(func, ToolNode)):
            # Name and function passed separately
            name = name_or_func
        else:
            raise ValueError("Invalid arguments for add_node")

        self.nodes[name] = Node(name, func)
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
    ) -> "StateGraph":
        """Add a static edge between two nodes.

        Creates a direct connection from one node to another. If the source
        node is START, the target node becomes the entry point for the graph.

        Args:
            from_node: Name of the source node.
            to_node: Name of the target node.

        Returns:
            StateGraph: The graph instance for method chaining.

        Example:
            >>> graph.add_edge("node1", "node2")
            >>> graph.add_edge(START, "entry_node")  # Sets entry point
        """
        # Set entry point if edge is from START
        if from_node == START:
            self.entry_point = to_node
        self.edges.append(Edge(from_node, to_node))
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        path_map: dict[str, str] | None = None,
    ) -> "StateGraph":
        """Add conditional edges from a node based on a condition function.

        Creates edges that are traversed based on the result of a condition
        function. The condition function receives the current state and should
        return a value that determines which edge to follow.

        Args:
            from_node: Name of the source node.
            condition: Function that evaluates the current state and returns
                a value to determine the next node.
            path_map: Optional mapping from condition results to target nodes.
                If provided, creates multiple conditional edges. If None,
                creates a single conditional edge.

        Returns:
            StateGraph: The graph instance for method chaining.

        Example:
            >>> def route_condition(state):
            ...     return "success" if state.success else "failure"
            >>> graph.add_conditional_edges(
            ...     "processor",
            ...     route_condition,
            ...     {"success": "next_step", "failure": "error_handler"},
            ... )
        """
        # Create edges based on possible returns from condition function
        if path_map:
            for condition_result, target_node in path_map.items():
                edge = Edge(from_node, target_node, condition)
                edge.condition_result = condition_result
                self.edges.append(edge)
        else:
            # Single conditional edge
            self.edges.append(Edge(from_node, "", condition))
        return self

    def add_sequence(
        self,
        nodes: list[str | Callable],
    ) -> "StateGraph":
        """Add a sequence of nodes with automatic edges."""
        processed_nodes = []

        for node in nodes:
            if callable(node):
                name = node.__name__
                self.add_node(name, node)
                processed_nodes.append(name)
            elif isinstance(node, str):
                processed_nodes.append(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")

        # Add edges between consecutive nodes
        for i in range(len(processed_nodes) - 1):
            self.add_edge(processed_nodes[i], processed_nodes[i + 1])

        return self

    def set_entry_point(self, node_name: str) -> "StateGraph":
        """Set the entry point for the graph."""
        self.entry_point = node_name
        self.add_edge(START, node_name)
        return self

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> "CompiledGraph[StateT]":
        """Compile the graph for execution.

        Args:
            checkpointer: Checkpointer for state persistence
            store: Store for additional data
            debug: Enable debug mode
            interrupt_before: List of node names to interrupt before execution
            interrupt_after: List of node names to interrupt after execution
            realtime_state_sync: Hook for frequent state sync (sync or async callable)
        """
        if not self.entry_point:
            raise GraphError("No entry point set. Use set_entry_point() or add an edge from START.")

        # Validate graph structure
        self._validate_graph()

        # Validate interrupt node names
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        all_interrupt_nodes = set(interrupt_before + interrupt_after)
        invalid_nodes = all_interrupt_nodes - set(self.nodes.keys())
        if invalid_nodes:
            raise GraphError(
                f"Invalid interrupt nodes: {invalid_nodes}. Must be existing node names."
            )

        self.compiled = True
        # Import here to avoid circular import at module import time

        # Import the CompiledGraph class
        from .compiled_graph import CompiledGraph

        return CompiledGraph(
            state_graph=self,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )

    def _validate_graph(self):
        """Validate the graph structure."""
        # Check for orphaned nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)

        all_nodes = set(self.nodes.keys())
        orphaned = all_nodes - connected_nodes
        if orphaned - {START, END}:  # START and END can be orphaned
            raise GraphError(f"Orphaned nodes detected: {orphaned - {START, END}}")

        # Check that all edge targets exist
        for edge in self.edges:
            if edge.to_node and edge.to_node not in self.nodes:
                raise GraphError(f"Edge targets non-existent node: {edge.to_node}")

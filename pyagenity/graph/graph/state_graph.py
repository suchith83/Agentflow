from typing import Any, Callable, Dict, Optional, Type, Union

from pyagenity.graph.exceptions.graph_error import GraphError
from pyagenity.graph.graph.compiled_graph import CompiledGraph
from pyagenity.graph.state.base_context import BaseContextManager
from pyagenity.graph.state.state import AgentState

from .edge import Edge

from .node import Node
from pyagenity.graph.utils.constants import START, END


class StateGraph:
    """
    Main graph class for orchestrating multi-agent workflows.

    Similar to LangGraph's StateGraph but designed for direct Litellm integration.
    """

    def __init__(
        self,
        state: AgentState = AgentState(),
        context_manager: Optional[BaseContextManager] = None,
    ):
        self.state = state
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: Optional[str] = None
        self.context_manager: Optional[BaseContextManager] = context_manager
        self.compiled = False

        # Add START and END nodes
        self.nodes[START] = Node(START, lambda s, c: s)
        self.nodes[END] = Node(END, lambda s, c: s)

    def add_node(
        self,
        name_or_func: Union[str, Callable],
        func: Optional[Callable] = None,
    ) -> "StateGraph":
        """Add a node to the graph."""
        if callable(name_or_func) and func is None:
            # Function passed as first argument
            name = name_or_func.__name__
            func = name_or_func
        elif isinstance(name_or_func, str) and callable(func):
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
        """Add a static edge between nodes."""
        # Set entry point if edge is from START
        if from_node == START:
            self.entry_point = to_node
        self.edges.append(Edge(from_node, to_node))
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        path_map: Optional[Dict[str, str]] = None,
    ) -> "StateGraph":
        """Add conditional edges from a node."""
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
        nodes: list[Union[str, Callable]],
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
        checkpointer: Optional[Any] = None,
        store: Optional[Any] = None,
        debug: bool = False,
    ) -> "CompiledGraph":
        """Compile the graph for execution."""
        if not self.entry_point:
            raise GraphError(
                "No entry point set. Use set_entry_point() or add an edge from START."
            )

        # Validate graph structure
        self._validate_graph()

        self.compiled = True
        return CompiledGraph(
            state_graph=self,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
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

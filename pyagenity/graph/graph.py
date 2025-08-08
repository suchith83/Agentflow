from __future__ import annotations
from typing import Dict, List, Any, Optional
from .edge import Edge
from .nodes import BaseNode


class Graph:
    """Directed conditional graph of nodes."""

    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[Edge] = []
        self.start_node: Optional[str] = None

    def add_node(self, node: BaseNode, start: bool = False) -> "Graph":
        self.nodes[node.name] = node
        if start or self.start_node is None:
            self.start_node = node.name
        return self

    def add_edge(self, edge: Edge) -> "Graph":
        self.edges.append(edge)
        return self

    def next_nodes(self, current: str, context: Dict[str, Any]) -> List[str]:
        out = []
        for e in self.edges:
            if e.source == current and e.is_triggered(context):
                out.append(e.target)
        return out

    def validate(self) -> None:
        if self.start_node is None:
            raise ValueError("Graph has no start node")
        for e in self.edges:
            if e.source not in self.nodes or e.target not in self.nodes:
                raise ValueError(f"Invalid edge references unknown node: {e}")

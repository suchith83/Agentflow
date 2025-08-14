from collections.abc import Callable
from typing import Optional


class Edge:
    """Represents an edge in the graph."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable] = None,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition  # For conditional edges
        self.condition_result: Optional[str] = None  # For mapped conditional edges

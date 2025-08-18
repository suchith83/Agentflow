from collections.abc import Callable


class Edge:
    """Represents an edge in the graph."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Callable | None = None,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition
        self.condition_result: str | None = None

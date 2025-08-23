import logging
from collections.abc import Callable


logger = logging.getLogger(__name__)


class Edge:
    """Represents an edge in the graph."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Callable | None = None,
    ):
        logger.debug(
            "Creating edge from '%s' to '%s' with condition=%s",
            from_node,
            to_node,
            "yes" if condition else "no",
        )
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition
        self.condition_result: str | None = None

from .graph_error import GraphError


class GraphRecursionError(GraphError):
    """Raised when graph execution exceeds recursion limit."""

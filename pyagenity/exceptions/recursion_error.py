import logging

from .graph_error import GraphError

logger = logging.getLogger(__name__)


class GraphRecursionError(GraphError):
    """Raised when graph execution exceeds recursion limit."""

    def __init__(self, message: str):
        logger.error("GraphRecursionError raised: %s", message)
        super().__init__(message)

import logging

from .graph_error import GraphError

logger = logging.getLogger(__name__)


class NodeError(GraphError):
    """Raised when a node encounters an error."""

    def __init__(self, message: str):
        logger.error("NodeError raised: %s", message)
        super().__init__(message)

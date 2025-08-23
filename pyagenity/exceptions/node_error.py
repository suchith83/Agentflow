import logging

from .graph_error import GraphError


logger = logging.getLogger(__name__)


class NodeError(GraphError):
    """
    Exception raised when a node encounters an error.

    This exception is used for errors specific to nodes within a graph.

    Example:
        >>> from pyagenity.exceptions.node_error import NodeError
        >>> raise NodeError("Node failed to execute")
    """

    def __init__(self, message: str):
        """
        Initializes a NodeError with the given message.

        Args:
            message (str): Description of the node error.
        """
        logger.error("NodeError raised: %s", message)
        super().__init__(message)

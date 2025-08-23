import logging

from .graph_error import GraphError


logger = logging.getLogger(__name__)


class GraphRecursionError(GraphError):
    """
    Exception raised when graph execution exceeds the recursion limit.

    This exception is used to indicate that a graph operation has recursed too deeply.

    Example:
        >>> from pyagenity.exceptions.recursion_error import GraphRecursionError
        >>> raise GraphRecursionError("Recursion limit exceeded in graph execution")
    """

    def __init__(self, message: str):
        """
        Initializes a GraphRecursionError with the given message.

        Args:
            message (str): Description of the recursion error.
        """
        logger.error("GraphRecursionError raised: %s", message)
        super().__init__(message)

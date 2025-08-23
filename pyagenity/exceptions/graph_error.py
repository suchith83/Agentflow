import logging


logger = logging.getLogger(__name__)


class GraphError(Exception):
    """
    Base exception for graph-related errors.

    This exception is raised when an error related to graph operations occurs.

    Example:
        >>> from pyagenity.exceptions.graph_error import GraphError
        >>> raise GraphError("Invalid graph structure")
    """

    def __init__(self, message: str):
        """
        Initializes a GraphError with the given message.

        Args:
            message (str): Description of the error.
        """
        logger.error("GraphError raised: %s", message)
        super().__init__(message)

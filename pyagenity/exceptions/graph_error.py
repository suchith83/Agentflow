import logging

logger = logging.getLogger(__name__)


class GraphError(Exception):
    """Base exception for graph-related errors."""

    def __init__(self, message: str):
        logger.error("GraphError raised: %s", message)
        super().__init__(message)

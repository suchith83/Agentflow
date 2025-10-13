"""
Centralized logging configuration for TAF.

This module provides logging configuration that can be imported and used
throughout the project. Each module should use:

    import logging
    logger = logging.getLogger(__name__)

This ensures proper hierarchical logging with module-specific loggers.

Typical usage example:
    >>> from agentflow.utils.logging import configure_logging
    >>> configure_logging(level=logging.DEBUG)

Functions:
    configure_logging: Configures the root logger for the TAF project.
"""

import logging
import sys


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configures the root logger for the TAF project.

    This function sets up logging for all modules under the 'agentflow' namespace.
    It ensures that logs are formatted consistently and sent to the appropriate handler.

    Args:
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.
        format_string (str, optional): Custom format string for log messages.
            If None, uses a default format: "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s".
        handler (logging.Handler, optional): Custom logging handler. If None,
            uses StreamHandler to stdout.

    Returns:
        None

    Raises:
        None

    Example:
        >>> configure_logging(level=logging.DEBUG)
        >>> logger = logging.getLogger("agentflow.module")
        >>> logger.info("This is an info message.")
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Configure root logger for agentflow
    root_logger = logging.getLogger("agentflow")
    root_logger.setLevel(level)

    # Only add handler if none exists to avoid duplicates
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


# Initialize default logging configuration
configure_logging()

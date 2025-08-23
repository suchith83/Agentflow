"""
Centralized logging configuration for PyAgenity.

This module provides logging configuration that can be imported and used
throughout the project. Each module should use:

    import logging
    logger = logging.getLogger(__name__)

This ensures proper hierarchical logging with module-specific loggers.
"""

import logging
import sys


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configure logging for the PyAgenity project.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        handler: Custom handler (default: StreamHandler to stdout)
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Configure root logger for pyagenity
    root_logger = logging.getLogger("pyagenity")
    root_logger.setLevel(level)

    # Only add handler if none exists to avoid duplicates
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


# Initialize default logging configuration
configure_logging()

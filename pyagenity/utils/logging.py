"""Simple logging framework for PyAgenity.

This module provides basic logging with debug, info, and error levels.
"""

import logging
import sys
from typing import Any


class PyAgenityLogger:
    """Simple logger for PyAgenity with debug, info, and error levels.

    This logger provides basic logging functionality without complex features
    like performance monitoring or security event logging.

    Attributes:
        logger (logging.Logger): The underlying Python logger instance.

    Example:
        >>> logger = PyAgenityLogger("my_module")
        >>> logger.debug("This is a debug message")
        >>> logger.info("This is an info message")
        >>> logger.error("This is an error message")
    """

    def __init__(self, name: str = "pyagenity"):
        """Initialize the logger with a given name.

        Args:
            name: Name for the logger. Defaults to "pyagenity".
        """
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up the logger with basic configuration."""
        # Set default level to INFO
        self.logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(handler)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message: The message to log.
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            message: The message to log.
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message: The message to log.
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message: The message to log.
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.error(message, *args, **kwargs)

    def set_level(self, level: str) -> None:
        """Set the logging level.

        Args:
            level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        self.logger.setLevel(numeric_level)


def get_logger(name: str = "pyagenity") -> PyAgenityLogger:
    """Get a logger instance.

    Args:
        name: Name for the logger. Defaults to "pyagenity".

    Returns:
        PyAgenityLogger: A configured logger instance.

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Application started")
    """
    return PyAgenityLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the entire PyAgenity framework.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
            Defaults to "INFO".
    """
    # Set the root pyagenity logger level
    root_logger = logging.getLogger("pyagenity")
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    root_logger.setLevel(numeric_level)


# Default logger instance for easy access
logger = get_logger()

"""
Logging utilities for Agentflow.

This module provides logging support for the Agentflow library following Python
logging best practices for library code.

By default, Agentflow uses a NullHandler to prevent "No handlers could be found"
warnings. Users can configure logging by getting the logger and adding their own
handlers.

Library Usage (within agentflow modules):
    Each module should create its own logger:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")

User Configuration Example:
    Users of the Agentflow library can configure logging like this::

        import logging

        # Configure the agentflow logger
        logger = logging.getLogger("agentflow")
        logger.setLevel(logging.DEBUG)

        # Add a handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

Best Practices:
    - Library code should NEVER configure the root logger
    - Library code should NEVER add handlers except NullHandler
    - Library code should use module-level loggers (logging.getLogger(__name__))
    - Users control logging configuration in their applications

References:
    https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
"""

import logging


# Create the main agentflow logger
logger = logging.getLogger("agentflow")

# Add NullHandler by default to prevent "No handlers found" warnings
# Users can configure their own handlers as needed
logger.addHandler(logging.NullHandler())

__all__ = [
    "logger",
]

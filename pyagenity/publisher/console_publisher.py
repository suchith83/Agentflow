"""Console publisher implementation for debugging and testing.

This module provides a publisher that outputs events to the console for development
and debugging purposes.
"""

import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger(__name__)


class ConsolePublisher(BasePublisher):
    """Publisher that prints events to the console for debugging and testing.

    This publisher is useful for development and debugging purposes, as it outputs event information
    to the standard output.

    Attributes:
        format: Output format ('json' by default).
        include_timestamp: Whether to include timestamp (True by default).
        indent: Indentation for output (2 by default).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the ConsolePublisher with the given configuration.

        Args:
            config: Configuration dictionary. Supported keys:
                - format: Output format (default: 'json').
                - include_timestamp: Whether to include timestamp (default: True).
                - indent: Indentation for output (default: 2).
        """
        super().__init__(config or {})
        self.format = config.get("format", "json") if config else "json"
        self.include_timestamp = config.get("include_timestamp", True) if config else True
        self.indent = config.get("indent", 2) if config else 2

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to the console.

        Args:
            event: The event to publish.

        Returns:
            None
        """
        msg = f"{event.timestamp} -> Source: {event.node_name}.{event.event_type}:"
        msg += f"-> Payload: {event.data}"
        msg += f" -> {event.metadata}"
        print(msg)  # noqa: T201

    async def close(self):
        """Close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for
        interface compatibility.
        """
        logger.debug("ConsolePublisher closed")

    def sync_close(self):
        """Synchronously close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for
        interface compatibility.
        """
        logger.debug("ConsolePublisher sync closed")

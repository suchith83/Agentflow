"""Console publisher implementation for debugging and testing."""

import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import Event


logger = logging.getLogger(__name__)


class ConsolePublisher(BasePublisher):
    """
    Publisher that prints events to the console for debugging and testing.

    This publisher is useful for development and debugging purposes, as it outputs event information
    to the standard output.

    Example:
        >>> from pyagenity.publisher.console_publisher import ConsolePublisher
        >>> from pyagenity.publisher.events import Event, SourceType, EventType
        >>> pub = ConsolePublisher({"format": "json"})
        >>> event = Event(source=SourceType.MESSAGE, event_type=EventType.INVOKED)
        >>> import asyncio
        >>> asyncio.run(pub.publish(event))
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ConsolePublisher with the given configuration.

        Args:
            config (dict[str, Any] | None): Configuration dictionary. Supported keys:
                - format (str): Output format (default: 'json').
                - include_timestamp (bool): Whether to include timestamp (default: True).
                - indent (int): Indentation for output (default: 2).
        """
        super().__init__(config or {})
        self.format = config.get("format", "json") if config else "json"
        self.include_timestamp = config.get("include_timestamp", True) if config else True
        self.indent = config.get("indent", 2) if config else 2

    async def publish(self, event: Event) -> Any:
        """
        Publish an event to the console.

        Args:
            event (Event): The event to publish.

        Returns:
            Any: The result of the publish operation (None).
        """
        msg = f"{event.timestamp} -> Source: {event.source}.{event.event_type}:"
        msg += f"-> Payload: {event.payload}"
        msg += f" -> {event.config} and {event.meta}"
        print("msg")  # noqa: T201

    def close(self):
        """
        Close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for interface compatibility.
        """
        logger.debug("ConsolePublisher closed")

    def sync_close(self):
        """
        Synchronously close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for interface compatibility.
        """
        logger.debug("ConsolePublisher sync closed")

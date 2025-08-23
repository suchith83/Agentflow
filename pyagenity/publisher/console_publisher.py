"""Console publisher implementation for debugging and testing."""

import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import Event


logger = logging.getLogger(__name__)


class ConsolePublisher(BasePublisher):
    """A publisher that prints events to the console."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.format = config.get("format", "json") if config else "json"
        self.include_timestamp = config.get("include_timestamp", True) if config else True
        self.indent = config.get("indent", 2) if config else 2

    async def publish(self, event: Event) -> Any:
        """Publish event to console."""
        msg = f"{event.timestamp} -> Source: {event.source}.{event.event_type}:"
        msg += f"-> Payload: {event.payload}"
        msg += f" -> {event.config} and {event.meta}"
        print("msg")  # noqa: T201

    def close(self):
        """Console publisher doesn't need cleanup."""
        logger.debug("ConsolePublisher closed")

    def sync_close(self):
        """Console publisher doesn't need cleanup."""
        logger.debug("ConsolePublisher sync closed")

"""Kafka publisher implementation (optional dependency).

Uses aiokafka to publish events to a Kafka topic.

Dependency: aiokafka
Not installed by default; install extra: `pip install pyagenity[kafka]`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger(__name__)


class KafkaPublisher(BasePublisher):
    """Publish events to a Kafka topic using aiokafka.

    This class provides an asynchronous interface for publishing events to a Kafka topic.
    It uses the aiokafka library to handle the producer operations. The publisher is
    lazily initialized and can be reused for multiple publishes.

    Example:
        >>> import asyncio
        >>> from pyagenity.publisher.kafka_publisher import KafkaPublisher
        >>> from pyagenity.models import EventModel
        >>> async def main():
        ...     config = {
        ...         "bootstrap_servers": "localhost:9092",
        ...         "topic": "my_topic",
        ...         "client_id": "my_client",
        ...     }
        ...     publisher = KafkaPublisher(config)
        ...     event = EventModel.stream({})
        ...     await publisher.publish(event)
        ...     await publisher.close()
        >>> asyncio.run(main())
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the KafkaPublisher.

        Args:
            config (dict[str, Any] | None): Configuration dictionary. Defaults to None.
                Supported keys:
                - bootstrap_servers (str): Kafka bootstrap servers. Defaults to "localhost:9092".
                - topic (str): Kafka topic to publish to. Defaults to "pyagenity.events".
                - client_id (str | None): Client ID for the producer. Defaults to None.
        """
        super().__init__(config or {})
        self.bootstrap_servers: str = self.config.get("bootstrap_servers", "localhost:9092")
        self.topic: str = self.config.get("topic", "pyagenity.events")
        self.client_id: str | None = self.config.get("client_id")
        self._producer = None  # type: ignore[var-annotated]

    async def _get_producer(self):
        """Get or create the Kafka producer instance.

        This method lazily initializes the producer if it hasn't been created yet.
        It imports aiokafka and starts the producer.

        Returns:
            aiokafka.AIOKafkaProducer: The initialized producer instance.

        Raises:
            RuntimeError: If the 'aiokafka' package is not installed.
        """
        if self._producer is not None:
            return self._producer

        try:
            aiokafka = importlib.import_module("aiokafka")
        except Exception as exc:
            raise RuntimeError(
                "KafkaPublisher requires the 'aiokafka' package. Install with "
                "'pip install pyagenity[kafka]' or 'pip install aiokafka'."
            ) from exc

        producer_cls = aiokafka.AIOKafkaProducer
        self._producer = producer_cls(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
        )
        await self._producer.start()
        return self._producer

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to the Kafka topic.

        Args:
            event (EventModel): The event to publish.

        Returns:
            Any: The result of the send_and_wait operation.
        """
        producer = await self._get_producer()
        payload = json.dumps(event.model_dump()).encode("utf-8")
        return await producer.send_and_wait(self.topic, payload)

    async def close(self):
        """Close the Kafka producer.

        Stops the producer and cleans up resources. Errors during stopping are logged
        but do not raise exceptions.
        """
        if self._producer is None:
            return

        try:
            await self._producer.stop()
        except Exception:
            logger.debug("KafkaPublisher close encountered an error", exc_info=True)
        finally:
            self._producer = None

    def sync_close(self):
        """Synchronously close the Kafka producer.

        This method runs the async close in a new event loop. If called within an
        active event loop, it logs a warning and skips the operation.
        """
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")

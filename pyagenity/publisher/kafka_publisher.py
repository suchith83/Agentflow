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

from pyagenity.utils.streaming import EventModel

from .base_publisher import BasePublisher


logger = logging.getLogger(__name__)


class KafkaPublisher(BasePublisher):
    """Publish events to a Kafka topic using aiokafka."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.bootstrap_servers: str = self.config.get("bootstrap_servers", "localhost:9092")
        self.topic: str = self.config.get("topic", "pyagenity.events")
        self.client_id: str | None = self.config.get("client_id")
        self._producer = None  # type: ignore[var-annotated]

    async def _get_producer(self):
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
        producer = await self._get_producer()
        payload = json.dumps(event.model_dump()).encode("utf-8")
        return await producer.send_and_wait(self.topic, payload)

    async def close(self):
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                logger.debug("KafkaPublisher close encountered an error", exc_info=True)
            finally:
                self._producer = None

    def sync_close(self):
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")

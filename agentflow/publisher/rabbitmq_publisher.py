"""RabbitMQ publisher implementation (optional dependency).

Uses aio-pika to publish events to an exchange with a routing key.

Dependency: aio-pika
Not installed by default; install extra: `pip install 10xscale-agentflow[rabbitmq]`.
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


class RabbitMQPublisher(BasePublisher):
    """Publish events to RabbitMQ using aio-pika.

    Attributes:
        url: RabbitMQ connection URL.
        exchange: Exchange name.
        routing_key: Routing key for messages.
        exchange_type: Type of exchange.
        declare: Whether to declare the exchange.
        durable: Whether the exchange is durable.
        _conn: Connection instance.
        _channel: Channel instance.
        _exchange: Exchange instance.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the RabbitMQPublisher.

        Args:
            config: Configuration dictionary. Supported keys:
                - url: RabbitMQ URL (default: "amqp://guest:guest@localhost/").
                - exchange: Exchange name (default: "agentflow.events").
                - routing_key: Routing key (default: "agentflow.events").
                - exchange_type: Exchange type (default: "topic").
                - declare: Whether to declare exchange (default: True).
                - durable: Whether exchange is durable (default: True).
        """
        super().__init__(config or {})
        self.url: str = self.config.get("url", "amqp://guest:guest@localhost/")
        self.exchange: str = self.config.get("exchange", "agentflow.events")
        self.routing_key: str = self.config.get("routing_key", "agentflow.events")
        self.exchange_type: str = self.config.get("exchange_type", "topic")
        self.declare: bool = self.config.get("declare", True)
        self.durable: bool = self.config.get("durable", True)

        self._conn = None  # type: ignore[var-annotated]
        self._channel = None  # type: ignore[var-annotated]
        self._exchange = None  # type: ignore[var-annotated]

    async def _ensure(self):
        """Ensure the connection, channel, and exchange are initialized."""
        if self._exchange is not None:
            return

        try:
            aio_pika = importlib.import_module("aio_pika")
        except Exception as exc:
            raise RuntimeError(
                "RabbitMQPublisher requires the 'aio-pika' package. Install with "
                "'pip install 10xscale-agentflow[rabbitmq]' or 'pip install aio-pika'."
            ) from exc

        # Connect and declare exchange if needed
        self._conn = await aio_pika.connect_robust(self.url)
        self._channel = await self._conn.channel()

        if self.declare:
            ex_type = getattr(
                aio_pika.ExchangeType,
                self.exchange_type.upper(),
                aio_pika.ExchangeType.TOPIC,
            )
            self._exchange = await self._channel.declare_exchange(
                self.exchange, ex_type, durable=self.durable
            )
        else:
            # Fall back to default exchange
            self._exchange = self._channel.default_exchange

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to RabbitMQ.

        Args:
            event: The event to publish.

        Returns:
            True on success.
        """
        await self._ensure()
        payload = json.dumps(event.model_dump()).encode("utf-8")

        aio_pika = importlib.import_module("aio_pika")
        message = aio_pika.Message(body=payload)
        if self._exchange is None:
            raise RuntimeError("RabbitMQPublisher exchange not initialized")
        await self._exchange.publish(message, routing_key=self.routing_key)
        return True

    async def close(self):
        """Close the RabbitMQ connection and channel."""
        try:
            if self._channel is not None:
                await self._channel.close()
        except Exception:
            logger.debug("RabbitMQPublisher channel close error", exc_info=True)
        finally:
            self._channel = None

        try:
            if self._conn is not None:
                await self._conn.close()
        except Exception:
            logger.debug("RabbitMQPublisher connection close error", exc_info=True)
        finally:
            self._conn = None
            self._exchange = None

    def sync_close(self):
        """Synchronously close the RabbitMQ connection."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")

"""Redis publisher implementation (optional dependency).

This publisher uses the redis-py asyncio client to publish events via:
- Pub/Sub channels (default), or
- Redis Streams (XADD) when configured with mode="stream".

Dependency: redis>=4.2 (provides redis.asyncio).
Not installed by default; install extra: `pip install pyagenity[redis]`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any

from pyagenity.publisher.events import EventModel

from .base_publisher import BasePublisher


logger = logging.getLogger(__name__)


class RedisPublisher(BasePublisher):
    """Publish events to Redis via Pub/Sub channel or Stream.

    Config keys (all optional):
    - url: Redis URL (default: "redis://localhost:6379/0")
    - mode: "pubsub" | "stream" (default: "pubsub")
    - channel: Pub/Sub channel name (default: "pyagenity.events")
    - stream: Stream name for XADD (default: "pyagenity.events")
    - maxlen: For streams, approximate maxlen (int, optional)
    - encoding: JSON encoding (default: "utf-8")
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.url: str = self.config.get("url", "redis://localhost:6379/0")
        self.mode: str = self.config.get("mode", "pubsub")
        self.channel: str = self.config.get("channel", "pyagenity.events")
        self.stream: str = self.config.get("stream", "pyagenity.events")
        self.maxlen: int | None = self.config.get("maxlen")
        self.encoding: str = self.config.get("encoding", "utf-8")

        # Lazy import & connect on first use to avoid ImportError at import-time.
        self._redis = None  # type: ignore[var-annotated]

    async def _get_client(self):
        if self._redis is not None:
            return self._redis

        try:
            redis_asyncio = importlib.import_module("redis.asyncio")
        except Exception as exc:  # ImportError and others
            raise RuntimeError(
                "RedisPublisher requires the 'redis' package. Install with "
                "'pip install pyagenity[redis]' or 'pip install redis'."
            ) from exc

        try:
            self._redis = redis_asyncio.from_url(
                self.url, encoding=self.encoding, decode_responses=False
            )
        except Exception as exc:
            raise RuntimeError(f"RedisPublisher failed to connect to Redis at {self.url}") from exc

        return self._redis

    async def publish(self, event: EventModel) -> Any:
        client = await self._get_client()
        payload = json.dumps(event.model_dump()).encode(self.encoding)

        if self.mode == "stream":
            # XADD to stream
            fields = {"data": payload}
            if self.maxlen is not None:
                return await client.xadd(self.stream, fields, maxlen=self.maxlen, approximate=True)
            return await client.xadd(self.stream, fields)

        # Default: Pub/Sub channel
        return await client.publish(self.channel, payload)

    async def close(self):
        if self._redis is not None:
            try:
                await self._redis.close()
                await self._redis.connection_pool.disconnect(inuse_connections=True)
            except Exception:  # best-effort close
                logger.debug("RedisPublisher close encountered an error", exc_info=True)
            finally:
                self._redis = None

    def sync_close(self):
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # Already in an event loop; fall back to scheduling close
            logger.warning("sync_close called within an active event loop; skipping.")

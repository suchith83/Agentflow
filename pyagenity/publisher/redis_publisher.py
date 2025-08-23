# from __future__ import annotations

# from __future__ import annotations

# import asyncio
# import json
# from typing import Any, Optional

# import redis.asyncio as aioredis

# from .base_publisher import BasePublisher


# class RedisPublisher(BasePublisher):
# 	"""Asynchronous Redis publisher.

# 	Config keys (in `config` passed to constructor):
# 	  - url: redis connection URL (default: redis://localhost:6379/0)
# 	  - channel: default channel to publish to (default: 'pyagenity')
# 	  - encoding: encoding for messages (default: 'utf-8')
# 	"""

# 	def __init__(self, config: dict[str, Any]) -> None:
# 		super().__init__(config)
# 		self._url: str = config.get("url", "redis://localhost:6379/0")
# 		self._channel: str = config.get("channel", "pyagenity")
# 		self._encoding: str = config.get("encoding", "utf-8")

# 		# Lazy-created async Redis client
# 		self._redis: Optional[aioredis.Redis] = None
# 		self._client_lock = asyncio.Lock()

# 	async def _get_client(self) -> aioredis.Redis:
# 		if self._redis is None:
# 			# ensure only one coroutine creates the client
# 			from __future__ import annotations

# 			import asyncio
# 			import json
# 			from typing import Any, Optional

# 			import redis.asyncio as aioredis

# 			from .base_publisher import BasePublisher


# 			class RedisPublisher(BasePublisher):
# 				"""Asynchronous Redis publisher.

# 				Config keys (in `config` passed to constructor):
# 				  - url: redis connection URL (default: redis://localhost:6379/0)
# 				  - channel: default channel to publish to (default: 'pyagenity')
# 				  - encoding: encoding for messages (default: 'utf-8')
# 				"""

# 				def __init__(self, config: dict[str, Any]) -> None:
# 					super().__init__(config)
# 					self._url: str = config.get("url", "redis://localhost:6379/0")
# 					self._channel: str = config.get("channel", "pyagenity")
# 					self._encoding: str = config.get("encoding", "utf-8")

# 					# Lazy-created async Redis client
# 					self._redis: Optional[aioredis.Redis] = None
# 					self._client_lock = asyncio.Lock()

# 				async def _get_client(self) -> aioredis.Redis:
# 					if self._redis is None:
# 						# ensure only one coroutine creates the client
# 						async with self._client_lock:
# 							if self._redis is None:
# 								# from_url returns an async Redis client
# 								self._redis = aioredis.from_url(
# 									self._url, encoding=self._encoding, decode_responses=True
# 								)
# 					return self._redis

# 				async def publish(self, info: dict[str, Any], channel: Optional[str] = None) -> int:
# 					"""Publish `info` to Redis channel asynchronously.

# 					Args:
# 						info: dictionary payload to serialize as JSON.
# 						channel: optional channel override; if omitted uses configured default.

# 					Returns:
# 						The number of clients that received the message (as returned by Redis PUBLISH).
# 					"""
# 					client = await self._get_client()
# 					payload = json.dumps(info, default=str)
# 					ch = channel or self._channel
# 					# publish returns number of subscribers that received the message
# 					return await client.publish(ch, payload)

# 				async def close(self) -> None:
# 					"""Close the underlying Redis client and connection pool."""
# 					if self._redis is None:
# 						return
# 					try:
# 						await self._redis.close()
# 						# ensure connections are disconnected
# 						if hasattr(self._redis, "connection_pool") and self._redis.connection_pool is not None:
# 							await self._redis.connection_pool.disconnect()
# 					finally:
# 						self._redis = None


# 			__all__ = ["RedisPublisher"]

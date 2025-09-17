"""Publisher module for PyAgenity events.

Includes optional publishers (Redis, Kafka, RabbitMQ) exposed only if their
dependencies are installed.
"""

from __future__ import annotations

import importlib

from .base_publisher import BasePublisher
from .console_publisher import ConsolePublisher


__all__ = ["BasePublisher", "ConsolePublisher"]


# Optional publishers
def _try_import(name: str, attr: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, attr)
    except Exception:
        return None


def _is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


RedisPublisher = None
KafkaPublisher = None
RabbitMQPublisher = None

if _is_available("redis.asyncio"):
    RedisPublisher = _try_import("pyagenity.publisher.redis_publisher", "RedisPublisher")
if _is_available("aiokafka"):
    KafkaPublisher = _try_import("pyagenity.publisher.kafka_publisher", "KafkaPublisher")
if _is_available("aio_pika"):
    RabbitMQPublisher = _try_import("pyagenity.publisher.rabbitmq_publisher", "RabbitMQPublisher")

if RedisPublisher:
    __all__.append("RedisPublisher")
if KafkaPublisher:
    __all__.append("KafkaPublisher")
if RabbitMQPublisher:
    __all__.append("RabbitMQPublisher")

"""Publisher module for PyAgenity events.

This module provides publishers that handle the delivery of events to various outputs,
such as console, Redis, Kafka, and RabbitMQ. Publishers are primarily used for
logging and monitoring agent behavior, enabling real-time tracking of performance,
usage, and debugging in agent graphs.

Key components:
- BasePublisher: Abstract base class for all publishers, defining the interface for publishing event
- ConsolePublisher: Default publisher that outputs events to the console for development
    and debugging
- Optional publishers: RedisPublisher, KafkaPublisher, RabbitMQPublisher, which are available
    only if their dependencies are installed

Usage:
- Import publishers: from pyagenity.publisher import ConsolePublisher, RedisPublisher (if available)
- Instantiate and use in CompiledGraph: graph.compile(publisher=ConsolePublisher()).
- Events are emitted as EventModel instances during graph execution, including node starts,
    completions, and errors.

Dependencies for optional publishers:
- RedisPublisher: Requires 'redis.asyncio' (install via pip install redis).
- KafkaPublisher: Requires 'aiokafka' (install via pip install aiokafka).
- RabbitMQPublisher: Requires 'aio_pika' (install via pip install aio-pika).

For more details, see the individual publisher classes and the PyAgenity documentation.
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

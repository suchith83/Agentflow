"""Publisher module for TAF events.

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
- Import publishers: from agentflow.publisher import ConsolePublisher, RedisPublisher (if available)
- Instantiate and use in CompiledGraph: graph.compile(publisher=ConsolePublisher()).
- Events are emitted as EventModel instances during graph execution, including node starts,
    completions, and errors.

Dependencies for optional publishers:
- RedisPublisher: Requires 'redis.asyncio' (install via pip install redis).
- KafkaPublisher: Requires 'aiokafka' (install via pip install aiokafka).
- RabbitMQPublisher: Requires 'aio_pika' (install via pip install aio-pika).

For more details, see the individual publisher classes and the TAF documentation.
"""

from .base_publisher import BasePublisher
from .console_publisher import ConsolePublisher
from .events import ContentType, Event, EventModel, EventType
from .kafka_publisher import KafkaPublisher
from .publish import publish_event
from .rabbitmq_publisher import RabbitMQPublisher
from .redis_publisher import RedisPublisher


__all__ = [
    "BasePublisher",
    "ConsolePublisher",
    "ContentType",
    "Event",
    "EventModel",
    "EventType",
    "KafkaPublisher",
    "RabbitMQPublisher",
    "RedisPublisher",
    "publish_event",
]

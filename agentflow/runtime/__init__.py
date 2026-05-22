"""Runtime components for Agentflow.

This package provides the runtime infrastructure for agent execution:

- ``agentflow.runtime.adapters``   - LLM response converters
- ``agentflow.runtime.publisher``  - event publishers (console, Redis, Kafka, RabbitMQ)
- ``agentflow.runtime.protocols``  - agent communication protocol packages
"""

from . import adapters, protocols, publisher
from .adapters.llm import (
    BaseConverter,
    ConverterType,
    GoogleGenAIConverter,
    OpenAIConverter,
    OpenAIResponsesConverter,
)
from .publisher import (
    BasePublisher,
    ConsolePublisher,
    ContentType,
    Event,
    EventModel,
    EventType,
    KafkaPublisher,
    RabbitMQPublisher,
    RedisPublisher,
    publish_event,
)


__all__ = [
    "AgentFlowExecutor",
    # Adapters
    "BaseConverter",
    "BasePublisher",
    "ConsolePublisher",
    "ContentType",
    "ConverterType",
    "Event",
    "EventModel",
    "EventType",
    "GoogleGenAIConverter",
    "KafkaPublisher",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
    "RabbitMQPublisher",
    "RedisPublisher",
    "a2a",
    "adapters",
    "build_a2a_app",
    "create_a2a_client_node",
    "create_a2a_server",
    "delegate_to_a2a_agent",
    "make_agent_card",
    # Protocols
    "protocols",
    "publish_event",
    "publisher",
]

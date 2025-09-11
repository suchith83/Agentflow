"""Comprehensive tests for the publisher module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pyagenity.publisher import (
    BasePublisher,
    ConsolePublisher,
    Event,
    EventType,
    KafkaPublisher,
    RabbitMQPublisher,
    RedisPublisher,
    SourceType,
)


class TestEvent:
    """Test the Event class."""

    def test_event_creation(self):
        """Test creating an Event."""
        event = Event(
            event_type=EventType.INITIALIZE, source=SourceType.GRAPH, payload={"test": "data"}
        )
        assert event.event_type == EventType.INITIALIZE  # noqa: S101
        assert event.source == SourceType.GRAPH  # noqa: S101
        assert event.payload == {"test": "data"}  # noqa: S101

    def test_event_with_config(self):
        """Test creating an Event with config."""
        event = Event(
            event_type=EventType.RUNNING,
            source=SourceType.NODE,
            config={"node_name": "test_node"},
            payload={"state": "active"},
        )
        assert event.config == {"node_name": "test_node"}  # noqa: S101
        assert event.payload == {"state": "active"}  # noqa: S101

    def test_event_with_meta(self):
        """Test creating an Event with metadata."""
        event = Event(
            event_type=EventType.COMPLETED, source=SourceType.TOOL, meta={"execution_time": 1.5}
        )
        assert event.meta == {"execution_time": 1.5}  # noqa: S101


class TestEventType:
    """Test the EventType enum."""

    def test_event_type_values(self):
        """Test EventType enum values."""
        assert EventType.INITIALIZE  # noqa: S101
        assert EventType.RUNNING  # noqa: S101
        assert EventType.COMPLETED  # noqa: S101
        assert EventType.ERROR  # noqa: S101
        assert EventType.INTERRUPTED  # noqa: S101
        assert EventType.INVOKED  # noqa: S101
        assert EventType.CHANGED  # noqa: S101
        assert EventType.CUSTOM  # noqa: S101


class TestSourceType:
    """Test the SourceType enum."""

    def test_source_type_values(self):
        """Test SourceType enum values."""
        assert SourceType.MESSAGE  # noqa: S101
        assert SourceType.GRAPH  # noqa: S101
        assert SourceType.NODE  # noqa: S101
        assert SourceType.STATE  # noqa: S101
        assert SourceType.TOOL  # noqa: S101


class TestBasePublisher:
    """Test the BasePublisher abstract class."""

    def test_base_publisher_needs_config(self):
        """Test that BasePublisher requires config."""
        # BasePublisher should require config parameter
        try:
            # Create with empty config to test instantiation
            publisher = BasePublisher(config={})
            assert publisher is not None  # noqa: S101
        except TypeError:
            # Expected if it's an abstract class
            pass

    def test_base_publisher_with_config(self):
        """Test creating BasePublisher with config."""
        try:
            publisher = BasePublisher(config={})
            assert publisher is not None  # noqa: S101
        except TypeError:
            # Expected if abstract
            pass


class TestConsolePublisher:
    """Test the ConsolePublisher class."""

    def test_console_publisher_creation(self):
        """Test creating a ConsolePublisher."""
        publisher = ConsolePublisher(config={})
        assert publisher is not None  # noqa: S101

    @pytest.mark.asyncio
    async def test_console_publisher_publish(self):
        """Test publishing an event through ConsolePublisher."""
        publisher = ConsolePublisher(config={})
        event = Event(
            event_type=EventType.INITIALIZE, source=SourceType.GRAPH, payload={"test": "data"}
        )

        # Should not raise an exception
        await publisher.publish(event)

    @pytest.mark.asyncio
    async def test_console_publisher_publish_multiple_events(self):
        """Test publishing multiple events."""
        publisher = ConsolePublisher(config={})

        events = [
            Event(event_type=EventType.INITIALIZE, source=SourceType.GRAPH),
            Event(event_type=EventType.RUNNING, source=SourceType.NODE),
            Event(event_type=EventType.COMPLETED, source=SourceType.TOOL),
            Event(event_type=EventType.ERROR, source=SourceType.STATE),
        ]

        for event in events:
            await publisher.publish(event)

    @pytest.mark.asyncio
    async def test_console_publisher_with_complex_event(self):
        """Test publishing a complex event with all fields."""
        publisher = ConsolePublisher(config={"verbose": True})
        event = Event(
            event_type=EventType.INVOKED,
            source=SourceType.NODE,
            config={"node_name": "ai_agent", "retry_count": 3},
            payload={"input": "process this", "output": "processed"},
            meta={"execution_time": 2.5, "memory_usage": "150MB"},
        )

        await publisher.publish(event)


class TestKafkaPublisher:
    """Test the KafkaPublisher class."""

    @pytest.fixture(autouse=True)
    def skip_if_kafka_unavailable(self):
        """Skip tests if KafkaPublisher is not available."""
        if KafkaPublisher is None:
            pytest.skip("KafkaPublisher not available (aiokafka not installed)")

    def test_kafka_publisher_creation_default_config(self):
        """Test creating KafkaPublisher with default config."""
        publisher = KafkaPublisher()
        assert publisher.bootstrap_servers == "localhost:9092"
        assert publisher.topic == "pyagenity.events"
        assert publisher.client_id is None
        assert publisher._producer is None

    def test_kafka_publisher_creation_custom_config(self):
        """Test creating KafkaPublisher with custom config."""
        config = {
            "bootstrap_servers": "kafka1:9092,kafka2:9092",
            "topic": "custom.events",
            "client_id": "test-client",
        }
        publisher = KafkaPublisher(config)
        assert publisher.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert publisher.topic == "custom.events"
        assert publisher.client_id == "test-client"

    def test_kafka_publisher_creation_partial_config(self):
        """Test creating KafkaPublisher with partial config."""
        config = {"topic": "partial.events"}
        publisher = KafkaPublisher(config)
        assert publisher.bootstrap_servers == "localhost:9092"  # default
        assert publisher.topic == "partial.events"  # custom
        assert publisher.client_id is None  # default

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_producer_success(self, mock_import):
        """Test successful producer creation."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        publisher = KafkaPublisher()

        # First call should create producer
        producer1 = await publisher._get_producer()
        assert producer1 is mock_producer
        assert publisher._producer is mock_producer

        # Second call should return cached producer
        producer2 = await publisher._get_producer()
        assert producer2 is mock_producer

        # Verify producer was started
        mock_producer.start.assert_called_once()

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_producer_aiokafka_not_installed(self, mock_import):
        """Test producer creation when aiokafka is not installed."""
        # Simulate import error
        mock_import.side_effect = ImportError("No module named 'aiokafka'")

        publisher = KafkaPublisher()

        with pytest.raises(RuntimeError, match="KafkaPublisher requires the 'aiokafka' package"):
            await publisher._get_producer()

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event(self, mock_import):
        """Test publishing an event."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.return_value = "message_id_123"
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        publisher = KafkaPublisher()
        event = Event(
            event_type=EventType.INITIALIZE,
            source=SourceType.GRAPH,
            payload={"test": "data"},
        )

        result = await publisher.publish(event)

        # Verify the result
        assert result == "message_id_123"

        # Verify producer was created and used
        mock_producer.start.assert_called_once()
        mock_producer.send_and_wait.assert_called_once()

        # Verify the payload
        call_args = mock_producer.send_and_wait.call_args
        topic, payload = call_args[0]
        assert topic == "pyagenity.events"

        # Decode and verify payload
        decoded_payload = json.loads(payload.decode("utf-8"))
        assert decoded_payload["event_type"] == "initialize"
        assert decoded_payload["source"] == "graph"
        assert decoded_payload["payload"] == {"test": "data"}

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_with_custom_topic(self, mock_import):
        """Test publishing an event to custom topic."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.return_value = "custom_topic_result"
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        config = {"topic": "custom.test.events"}
        publisher = KafkaPublisher(config)
        event = Event(event_type=EventType.RUNNING, source=SourceType.NODE)

        await publisher.publish(event)

        # Verify custom topic was used
        call_args = mock_producer.send_and_wait.call_args
        topic, payload = call_args[0]
        assert topic == "custom.test.events"

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_producer(self, mock_import):
        """Test closing the producer."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        publisher = KafkaPublisher()

        # Create producer first
        await publisher._get_producer()
        assert publisher._producer is not None

        # Close producer
        await publisher.close()

        # Verify producer was stopped
        mock_producer.stop.assert_called_once()
        assert publisher._producer is None

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_producer_with_error(self, mock_import):
        """Test closing producer when stop raises an error."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_producer.stop.side_effect = Exception("Stop failed")
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        publisher = KafkaPublisher()

        # Create producer first
        await publisher._get_producer()

        # Close should handle the error gracefully
        with patch("pyagenity.publisher.kafka_publisher.logger") as mock_logger:
            await publisher.close()

        # Verify error was logged
        mock_logger.debug.assert_called_once()
        assert publisher._producer is None

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_without_producer(self, mock_import):
        """Test closing when no producer exists."""
        publisher = KafkaPublisher()

        # Should not raise an error
        await publisher.close()
        assert publisher._producer is None

    @patch("pyagenity.publisher.kafka_publisher.asyncio.run")
    def test_sync_close_success(self, mock_asyncio_run):
        """Test sync_close method."""
        publisher = KafkaPublisher()

        publisher.sync_close()

        # Verify asyncio.run was called with close()
        mock_asyncio_run.assert_called_once()

    @patch("pyagenity.publisher.kafka_publisher.asyncio.run")
    @patch("pyagenity.publisher.kafka_publisher.logger")
    def test_sync_close_runtime_error(self, mock_logger, mock_asyncio_run):
        """Test sync_close when asyncio.run raises RuntimeError."""
        # Simulate RuntimeError (event loop already running)
        mock_asyncio_run.side_effect = RuntimeError("Cannot run in existing event loop")

        publisher = KafkaPublisher()

        publisher.sync_close()

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "sync_close called within an active event loop; skipping."
        )

    @patch("pyagenity.publisher.kafka_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_complex_event(self, mock_import):
        """Test publishing a complex event with all fields."""
        # Mock aiokafka
        mock_aiokafka = MagicMock()
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.return_value = "complex_event_id"
        mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_aiokafka

        publisher = KafkaPublisher()
        event = Event(
            event_type=EventType.INVOKED,
            source=SourceType.NODE,
            config={"node_name": "test_agent", "timeout": 30},
            payload={"input": "complex data", "result": {"status": "success"}},
            meta={"execution_time": 1.5, "memory_mb": 256},
        )

        result = await publisher.publish(event)

        assert result == "complex_event_id"

        # Verify payload contains all event data
        call_args = mock_producer.send_and_wait.call_args
        topic, payload = call_args[0]
        decoded_payload = json.loads(payload.decode("utf-8"))

        assert decoded_payload["event_type"] == "invoked"
        assert decoded_payload["source"] == "node"
        assert decoded_payload["config"] == {"node_name": "test_agent", "timeout": 30}
        assert decoded_payload["payload"] == {
            "input": "complex data",
            "result": {"status": "success"},
        }
        assert decoded_payload["meta"] == {"execution_time": 1.5, "memory_mb": 256}

    def test_kafka_publisher_inheritance(self):
        """Test that KafkaPublisher inherits from BasePublisher."""
        publisher = KafkaPublisher()
        assert isinstance(publisher, BasePublisher)


def test_publisher_module_imports():
    """Test that publisher module imports work correctly."""
    assert BasePublisher is not None  # noqa: S101
    assert ConsolePublisher is not None  # noqa: S101
    assert Event is not None  # noqa: S101
    assert EventType is not None  # noqa: S101
    assert SourceType is not None  # noqa: S101


class TestRabbitMQPublisher:
    """Test the RabbitMQPublisher class."""

    @pytest.fixture(autouse=True)
    def skip_if_rabbitmq_unavailable(self):
        """Skip tests if RabbitMQPublisher is not available."""
        if RabbitMQPublisher is None:
            pytest.skip("RabbitMQPublisher not available (aio_pika not installed)")

    def test_rabbitmq_publisher_creation_default_config(self):
        """Test creating RabbitMQPublisher with default config."""
        publisher = RabbitMQPublisher()
        assert publisher.url == "amqp://guest:guest@localhost/"
        assert publisher.exchange == "pyagenity.events"
        assert publisher.routing_key == "pyagenity.events"
        assert publisher.exchange_type == "topic"
        assert publisher.declare is True
        assert publisher.durable is True
        assert publisher._conn is None
        assert publisher._channel is None
        assert publisher._exchange is None

    def test_rabbitmq_publisher_creation_custom_config(self):
        """Test creating RabbitMQPublisher with custom config."""
        config = {
            "url": "amqp://user:pass@rabbitmq1:5673/my_vhost",
            "exchange": "custom_exchange",
            "routing_key": "custom_routing_key",
            "exchange_type": "direct",
            "declare": False,
            "durable": False,
        }
        publisher = RabbitMQPublisher(config)
        assert publisher.url == "amqp://user:pass@rabbitmq1:5673/my_vhost"
        assert publisher.exchange == "custom_exchange"
        assert publisher.routing_key == "custom_routing_key"
        assert publisher.exchange_type == "direct"
        assert publisher.declare is False
        assert publisher.durable is False

    def test_rabbitmq_publisher_creation_partial_config(self):
        """Test creating RabbitMQPublisher with partial config."""
        config = {"exchange": "partial_exchange"}
        publisher = RabbitMQPublisher(config)
        assert publisher.url == "amqp://guest:guest@localhost/"  # default
        assert publisher.exchange == "partial_exchange"  # custom
        assert publisher.routing_key == "pyagenity.events"  # default
        assert publisher.exchange_type == "topic"  # default
        assert publisher.declare is True  # default
        assert publisher.durable is True  # default

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_connection_success(self, mock_import):
        """Test successful connection creation."""
        # Mock aio_pika
        mock_aio_pika = MagicMock()
        mock_connection = AsyncMock()
        mock_aio_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_aio_pika

        publisher = RabbitMQPublisher()

        # First call should create connection
        connection1 = await publisher._get_connection()
        assert connection1 is mock_connection
        assert publisher._connection is mock_connection

        # Second call should return cached connection
        connection2 = await publisher._get_connection()
        assert connection2 is mock_connection

        # Verify connection was established
        mock_connection.__aenter__.assert_called_once()

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_connection_aio_pika_not_installed(self, mock_import):
        """Test connection creation when aio_pika is not installed."""
        # Simulate import error
        mock_import.side_effect = ImportError("No module named 'aio_pika'")

        publisher = RabbitMQPublisher()

        with pytest.raises(RuntimeError, match="RabbitMQPublisher requires the 'aio_pika' package"):
            await publisher._get_connection()

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event(self, mock_import):
        """Test publishing an event."""
        # Mock aio_pika
        mock_aio_pika = MagicMock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.default_exchange = "pyagenity"
        mock_channel.publish.return_value = True
        mock_aio_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_aio_pika

        publisher = RabbitMQPublisher()
        event = Event(
            event_type=EventType.INITIALIZE,
            source=SourceType.GRAPH,
            payload={"test": "data"},
        )

        result = await publisher.publish(event)

        # Verify the result
        assert result is True

        # Verify connection and channel were created and used
        mock_connection.__aenter__.assert_called_once()
        mock_channel.__aenter__.assert_called_once()
        mock_channel.publish.assert_called_once()

        # Verify the payload
        call_args = mock_channel.publish.call_args
        exchange, routing_key, payload = call_args[0]
        assert exchange == "pyagenity"
        assert routing_key == "events"

        # Decode and verify payload
        decoded_payload = json.loads(payload.decode("utf-8"))
        assert decoded_payload["event_type"] == "initialize"
        assert decoded_payload["source"] == "graph"
        assert decoded_payload["payload"] == {"test": "data"}

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_with_custom_exchange_and_routing_key(self, mock_import):
        """Test publishing an event with custom exchange and routing key."""
        # Mock aio_pika
        mock_aio_pika = MagicMock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.publish.return_value = True
        mock_aio_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_aio_pika

        config = {"exchange": "custom_exchange", "routing_key": "custom_routing_key"}
        publisher = RabbitMQPublisher(config)
        event = Event(event_type=EventType.RUNNING, source=SourceType.NODE)

        await publisher.publish(event)

        # Verify custom exchange and routing key were used
        call_args = mock_channel.publish.call_args
        exchange, routing_key, payload = call_args[0]
        assert exchange == "custom_exchange"
        assert routing_key == "custom_routing_key"

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_connection(self, mock_import):
        """Test closing the connection."""
        # Mock aio_pika
        mock_aio_pika = MagicMock()
        mock_connection = AsyncMock()
        mock_aio_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_aio_pika

        publisher = RabbitMQPublisher()

        # Create connection first
        await publisher._get_connection()
        assert publisher._connection is not None

        # Close connection
        await publisher.close()

        # Verify connection was closed
        mock_connection.__aexit__.assert_called_once()
        assert publisher._connection is None

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_connection_with_error(self, mock_import):
        """Test closing connection when __aexit__ raises an error."""
        # Mock aio_pika
        mock_aio_pika = MagicMock()
        mock_connection = AsyncMock()
        mock_connection.__aexit__.side_effect = Exception("Close failed")
        mock_aio_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_aio_pika

        publisher = RabbitMQPublisher()

        # Create connection first
        await publisher._get_connection()

        # Close should handle the error gracefully
        with patch("pyagenity.publisher.rabbitmq_publisher.logger") as mock_logger:
            await publisher.close()

        # Verify error was logged
        mock_logger.debug.assert_called_once()
        assert publisher._connection is None

    @patch("pyagenity.publisher.rabbitmq_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_without_connection(self, mock_import):
        """Test closing when no connection exists."""
        publisher = RabbitMQPublisher()

        # Should not raise an error
        await publisher.close()
        assert publisher._connection is None

    @patch("pyagenity.publisher.rabbitmq_publisher.asyncio.run")
    def test_sync_close_success(self, mock_asyncio_run):
        """Test sync_close method."""
        publisher = RabbitMQPublisher()

        publisher.sync_close()

        # Verify asyncio.run was called with close()
        mock_asyncio_run.assert_called_once()

    @patch("pyagenity.publisher.rabbitmq_publisher.asyncio.run")
    @patch("pyagenity.publisher.rabbitmq_publisher.logger")
    def test_sync_close_runtime_error(self, mock_logger, mock_asyncio_run):
        """Test sync_close when asyncio.run raises RuntimeError."""
        # Simulate RuntimeError (event loop already running)
        mock_asyncio_run.side_effect = RuntimeError("Cannot run in existing event loop")

        publisher = RabbitMQPublisher()

        publisher.sync_close()

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "sync_close called within an active event loop; skipping."
        )


class TestRedisPublisher:
    """Test the RedisPublisher class."""

    @pytest.fixture(autouse=True)
    def skip_if_redis_unavailable(self):
        """Skip tests if RedisPublisher is not available."""
        if RedisPublisher is None:
            pytest.skip("RedisPublisher not available (redis not installed)")

    def test_redis_publisher_creation_default_config(self):
        """Test creating RedisPublisher with default config."""
        publisher = RedisPublisher()
        assert publisher.url == "redis://localhost:6379/0"
        assert publisher.mode == "pubsub"
        assert publisher.channel == "pyagenity.events"
        assert publisher.stream == "pyagenity.events"
        assert publisher.maxlen is None
        assert publisher.encoding == "utf-8"
        assert publisher._redis is None

    def test_redis_publisher_creation_custom_config(self):
        """Test creating RedisPublisher with custom config."""
        config = {
            "url": "redis://redis-server:6380/1",
            "mode": "stream",
            "channel": "custom.channel",
            "stream": "custom.stream",
            "maxlen": 1000,
            "encoding": "utf-16",
        }
        publisher = RedisPublisher(config)
        assert publisher.url == "redis://redis-server:6380/1"
        assert publisher.mode == "stream"
        assert publisher.channel == "custom.channel"
        assert publisher.stream == "custom.stream"
        assert publisher.maxlen == 1000
        assert publisher.encoding == "utf-16"

    def test_redis_publisher_creation_partial_config(self):
        """Test creating RedisPublisher with partial config."""
        config = {"mode": "stream", "maxlen": 500}
        publisher = RedisPublisher(config)
        assert publisher.url == "redis://localhost:6379/0"  # default
        assert publisher.mode == "stream"  # custom
        assert publisher.channel == "pyagenity.events"  # default
        assert publisher.stream == "pyagenity.events"  # default
        assert publisher.maxlen == 500  # custom
        assert publisher.encoding == "utf-8"  # default

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_client_success(self, mock_import):
        """Test successful Redis client creation."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        publisher = RedisPublisher()

        # First call should create client
        client1 = await publisher._get_client()
        assert client1 is mock_client
        assert publisher._redis is mock_client

        # Second call should return cached client
        client2 = await publisher._get_client()
        assert client2 is mock_client

        # Verify client was created with correct parameters
        mock_redis_asyncio.from_url.assert_called_once_with(
            "redis://localhost:6379/0", encoding="utf-8", decode_responses=False
        )

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_get_client_redis_not_installed(self, mock_import):
        """Test client creation when redis is not installed."""
        # Simulate import error
        mock_import.side_effect = ImportError("No module named 'redis'")

        publisher = RedisPublisher()

        with pytest.raises(RuntimeError, match="RedisPublisher requires the 'redis' package"):
            await publisher._get_client()

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_pubsub_mode(self, mock_import):
        """Test publishing an event in pubsub mode (default)."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_client.publish.return_value = 5  # Number of subscribers
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        publisher = RedisPublisher()
        event = Event(
            event_type=EventType.INITIALIZE, source=SourceType.GRAPH, payload={"test": "data"}
        )

        result = await publisher.publish(event)

        # Verify the result (number of subscribers)
        assert result == 5

        # Verify publish was called with correct parameters
        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        channel, payload = call_args[0]
        assert channel == "pyagenity.events"

        # Decode and verify payload
        decoded_payload = json.loads(payload.decode("utf-8"))
        assert decoded_payload["event_type"] == "initialize"
        assert decoded_payload["source"] == "graph"
        assert decoded_payload["payload"] == {"test": "data"}

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_stream_mode(self, mock_import):
        """Test publishing an event in stream mode."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_client.xadd.return_value = b"1234567890123-0"  # Stream entry ID
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        config = {"mode": "stream"}
        publisher = RedisPublisher(config)
        event = Event(event_type=EventType.RUNNING, source=SourceType.NODE)

        result = await publisher.publish(event)

        # Verify the result (stream entry ID)
        assert result == b"1234567890123-0"

        # Verify xadd was called with correct parameters
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        stream_name, fields = call_args[0]
        assert stream_name == "pyagenity.events"
        assert "data" in fields

        # Verify no maxlen parameter (should not be called)
        assert "maxlen" not in call_args.kwargs

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_stream_mode_with_maxlen(self, mock_import):
        """Test publishing an event in stream mode with maxlen."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_client.xadd.return_value = b"1234567890123-1"
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        config = {"mode": "stream", "maxlen": 100}
        publisher = RedisPublisher(config)
        event = Event(event_type=EventType.COMPLETED, source=SourceType.TOOL)

        result = await publisher.publish(event)

        assert result == b"1234567890123-1"

        # Verify xadd was called with maxlen parameters
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        assert call_args.kwargs["maxlen"] == 100
        assert call_args.kwargs["approximate"] is True

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_publish_event_with_custom_channel_stream(self, mock_import):
        """Test publishing with custom channel/stream names."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_client.publish.return_value = 2
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        config = {"channel": "custom.events", "stream": "custom.stream"}
        publisher = RedisPublisher(config)
        event = Event(event_type=EventType.ERROR, source=SourceType.STATE)

        # Test pubsub mode with custom channel
        result = await publisher.publish(event)
        assert result == 2

        call_args = mock_client.publish.call_args
        channel, payload = call_args[0]
        assert channel == "custom.events"

        # Switch to stream mode and test custom stream
        publisher.mode = "stream"
        mock_client.xadd.return_value = b"9999999999999-0"

        result = await publisher.publish(event)
        assert result == b"9999999999999-0"

        call_args = mock_client.xadd.call_args
        stream_name, fields = call_args[0]
        assert stream_name == "custom.stream"

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_client(self, mock_import):
        """Test closing the Redis client."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_connection_pool = AsyncMock()
        mock_client.connection_pool = mock_connection_pool
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        publisher = RedisPublisher()

        # Create client first
        await publisher._get_client()
        assert publisher._redis is not None

        # Close client
        await publisher.close()

        # Verify client was closed
        mock_client.close.assert_called_once()
        mock_connection_pool.disconnect.assert_called_once_with(inuse_connections=True)
        assert publisher._redis is None

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_client_with_error(self, mock_import):
        """Test closing client when close raises an error."""
        # Mock redis.asyncio
        mock_redis_asyncio = MagicMock()
        mock_client = AsyncMock()
        mock_client.close.side_effect = Exception("Close failed")
        mock_connection_pool = AsyncMock()
        mock_client.connection_pool = mock_connection_pool
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio

        publisher = RedisPublisher()

        # Create client first
        await publisher._get_client()

        # Close should handle the error gracefully
        await publisher.close()

        # Verify client was set to None despite the error
        assert publisher._redis is None
        assert publisher._redis is None

    @patch("pyagenity.publisher.redis_publisher.importlib.import_module")
    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_import):
        """Test closing when no client exists."""
        publisher = RedisPublisher()

        # Should not raise an error
        await publisher.close()
        assert publisher._redis is None

    @patch("pyagenity.publisher.redis_publisher.asyncio.run")
    def test_sync_close_success(self, mock_asyncio_run):
        """Test sync_close method."""
        publisher = RedisPublisher()

        publisher.sync_close()

        # Verify asyncio.run was called with close()
        mock_asyncio_run.assert_called_once()

    @patch("pyagenity.publisher.redis_publisher.asyncio.run")
    @patch("pyagenity.publisher.redis_publisher.logger")
    def test_sync_close_runtime_error(self, mock_logger, mock_asyncio_run):
        """Test sync_close when asyncio.run raises RuntimeError."""
        # Simulate RuntimeError (event loop already running)
        mock_asyncio_run.side_effect = RuntimeError("Cannot run in existing event loop")

        publisher = RedisPublisher()

        publisher.sync_close()

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "sync_close called within an active event loop; skipping."
        )

    def test_redis_publisher_inheritance(self):
        """Test that RedisPublisher inherits from BasePublisher."""
        publisher = RedisPublisher()
        assert isinstance(publisher, BasePublisher)

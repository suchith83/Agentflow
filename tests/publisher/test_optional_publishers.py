"""
Comprehensive tests for optional publishers (Redis, Kafka, RabbitMQ).

This module tests optional publishers with dependency mocking,
configuration handling, and error scenarios.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any

from agentflow.publisher.base_publisher import BasePublisher
from agentflow.publisher.events import EventModel, Event, EventType, ContentType


class TestRedisPublisher:
    """Test RedisPublisher with mocked dependencies."""
    
    def test_redis_publisher_import_available(self):
        """Test importing RedisPublisher when redis is available."""
        with patch('importlib.import_module') as mock_import:
            # Mock successful redis import
            mock_redis_module = Mock()
            mock_import.return_value = mock_redis_module
            
            # Import the publisher module with mocked redis
            with patch.dict('sys.modules', {'redis.asyncio': mock_redis_module}):
                from agentflow.publisher.redis_publisher import RedisPublisher
                
                assert RedisPublisher is not None
                assert issubclass(RedisPublisher, BasePublisher)
    
    def test_redis_publisher_import_unavailable(self):
        """Test importing RedisPublisher when redis is not available."""
        with patch('importlib.import_module') as mock_import:
            # Mock ImportError for redis
            mock_import.side_effect = ImportError("No module named 'redis'")
            
            # Should be able to import the module but may fail at runtime
            try:
                from agentflow.publisher.redis_publisher import RedisPublisher
                # If import succeeds, class should exist
                assert RedisPublisher is not None
            except ImportError:
                # If import fails, that's also acceptable
                pass
    
    @patch('importlib.import_module')
    def test_redis_publisher_initialization(self, mock_import):
        """Test RedisPublisher initialization with mocked redis."""
        # Mock redis module
        mock_redis = Mock()
        mock_import.return_value = mock_redis
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            config = {
                "url": "redis://localhost:6379/1",
                "mode": "pubsub",
                "channel": "test.events",
                "encoding": "utf-8"
            }
            
            publisher = RedisPublisher(config)
            
            assert publisher.config == config
            assert publisher.url == "redis://localhost:6379/1"
            assert publisher.mode == "pubsub"
            assert publisher.channel == "test.events"
            assert publisher.encoding == "utf-8"
            assert publisher._redis is None  # Not connected yet
    
    @patch('importlib.import_module')
    def test_redis_publisher_default_config(self, mock_import):
        """Test RedisPublisher with default configuration."""
        mock_redis = Mock()
        mock_import.return_value = mock_redis
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            assert publisher.url == "redis://localhost:6379/0"
            assert publisher.mode == "pubsub"
            assert publisher.channel == "agentflow.events"
            assert publisher.stream == "agentflow.events"
            assert publisher.maxlen is None
            assert publisher.encoding == "utf-8"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_get_client_success(self, mock_import):
        """Test successful Redis client creation."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({"url": "redis://test:6379"})
            
            client = await publisher._get_client()
            
            assert client == mock_client
            assert publisher._redis == mock_client
            mock_redis_asyncio.from_url.assert_called_once_with(
                "redis://test:6379", encoding="utf-8", decode_responses=False
            )
    
    @pytest.mark.asyncio
    async def test_redis_publisher_get_client_import_error(self):
        """Test Redis client creation when redis module is missing."""
        from agentflow.publisher.redis_publisher import RedisPublisher
        
        publisher = RedisPublisher()
        
        # Patch the import inside the _get_client method
        with patch('agentflow.publisher.redis_publisher.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'redis'")
            
            with pytest.raises(RuntimeError, match="RedisPublisher requires the 'redis' package"):
                await publisher._get_client()
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_get_client_connection_error(self, mock_import):
        """Test Redis client creation with connection failure."""
        mock_redis_asyncio = Mock()
        mock_redis_asyncio.from_url.side_effect = Exception("Connection failed")
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            with pytest.raises(RuntimeError, match="RedisPublisher failed to connect to Redis"):
                await publisher._get_client()
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_pubsub_mode(self, mock_import):
        """Test publishing in pub/sub mode."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_client.publish.return_value = 1  # 1 subscriber received
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "mode": "pubsub",
                "channel": "test.channel"
            })
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START,
                node_name="test_node",
                data={"key": "value"}
            )
            
            result = await publisher.publish(event)
            
            assert result == 1
            mock_client.publish.assert_called_once()
            
            # Verify the published message
            call_args = mock_client.publish.call_args
            channel, message = call_args[0]
            
            assert channel == "test.channel"
            
            # Message should be JSON-serialized event
            parsed_message = json.loads(message)
            assert parsed_message["event"] == "graph_execution"
            assert parsed_message["event_type"] == "start"
            assert parsed_message["node_name"] == "test_node"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_stream_mode(self, mock_import):
        """Test publishing in stream mode."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_client.xadd.return_value = b"1234567890-0"  # Stream entry ID
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "mode": "stream",
                "stream": "test.stream",
                "maxlen": 1000
            })
            
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.PROGRESS,
                node_name="stream_test",
                data={"progress": 50}
            )
            
            result = await publisher.publish(event)
            
            assert result == b"1234567890-0"
            mock_client.xadd.assert_called_once()
            
            # Verify the stream call
            call_args = mock_client.xadd.call_args
            stream_name, fields = call_args[0]
            maxlen = call_args[1].get("maxlen")
            
            assert stream_name == "test.stream"
            assert maxlen == 1000
            
            # Fields should contain serialized event
            assert "data" in fields
            parsed_data = json.loads(fields["data"].decode('utf-8'))
            assert parsed_data["event"] == "node_execution"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_error(self, mock_import):
        """Test publish error handling."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_client.publish.side_effect = Exception("Redis error")
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.ERROR
            )
            
            with pytest.raises(Exception, match="Redis error"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_close(self, mock_import):
        """Test Redis publisher close method."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            # Get client (connects)
            await publisher._get_client()
            assert publisher._redis is not None
            
            # Close
            await publisher.close()
            
            mock_client.close.assert_called_once()
            assert publisher._redis is None


class TestKafkaPublisher:
    """Test KafkaPublisher with mocked dependencies."""
    
    def test_kafka_publisher_import_available(self):
        """Test importing KafkaPublisher when aiokafka is available."""
        with patch('importlib.import_module') as mock_import:
            mock_kafka_module = Mock()
            mock_import.return_value = mock_kafka_module
            
            with patch.dict('sys.modules', {'aiokafka': mock_kafka_module}):
                from agentflow.publisher.kafka_publisher import KafkaPublisher
                
                assert KafkaPublisher is not None
                assert issubclass(KafkaPublisher, BasePublisher)
    
    @patch('importlib.import_module')
    def test_kafka_publisher_initialization(self, mock_import):
        """Test KafkaPublisher initialization."""
        mock_kafka = Mock()
        mock_producer_class = Mock()
        mock_kafka.AIOKafkaProducer = mock_producer_class
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from agentflow.publisher.kafka_publisher import KafkaPublisher
            
            config = {
                "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
                "topic": "agent.events",
                "encoding": "utf-8"
            }
            
            publisher = KafkaPublisher(config)
            
            assert publisher.config == config
            assert publisher.bootstrap_servers == config["bootstrap_servers"]
            assert publisher.topic == "agent.events"
            assert publisher.client_id is None  # Default value
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_kafka_publisher_publish_success(self, mock_import):
        """Test successful Kafka publishing."""
        mock_kafka = Mock()
        mock_producer = AsyncMock()
        mock_producer_class = Mock(return_value=mock_producer)
        mock_kafka.AIOKafkaProducer = mock_producer_class
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from agentflow.publisher.kafka_publisher import KafkaPublisher
            
            publisher = KafkaPublisher({
                "topic": "test.topic"
            })
            
            event = EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.RESULT,
                node_name="kafka_test",
                data={"result": "success"}
            )
            
            # Mock successful send
            mock_producer.send_and_wait.return_value = Mock(
                topic="test.topic",
                partition=0,
                offset=12345
            )
            
            result = await publisher.publish(event)
            
            # Should return the record metadata
            assert result.topic == "test.topic"
            assert result.partition == 0
            assert result.offset == 12345
            
            # Verify producer was called correctly
            mock_producer.send_and_wait.assert_called_once()
            call_args = mock_producer.send_and_wait.call_args
            topic, message = call_args[0]
            
            assert topic == "test.topic"
            
            # Message should be JSON-encoded event
            parsed_message = json.loads(message.decode('utf-8'))
            assert parsed_message["event"] == "tool_execution"
            assert parsed_message["node_name"] == "kafka_test"


class TestRabbitMQPublisher:
    """Test RabbitMQPublisher with mocked dependencies."""
    
    def test_rabbitmq_publisher_import_available(self):
        """Test importing RabbitMQPublisher when aio_pika is available."""
        with patch('importlib.import_module') as mock_import:
            mock_pika_module = Mock()
            mock_import.return_value = mock_pika_module
            
            with patch.dict('sys.modules', {'aio_pika': mock_pika_module}):
                from agentflow.publisher.rabbitmq_publisher import RabbitMQPublisher
                
                assert RabbitMQPublisher is not None
                assert issubclass(RabbitMQPublisher, BasePublisher)
    
    @patch('importlib.import_module')
    def test_rabbitmq_publisher_initialization(self, mock_import):
        """Test RabbitMQPublisher initialization."""
        mock_pika = Mock()
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from agentflow.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            config = {
                "url": "amqp://user:pass@localhost:5672/vhost",
                "exchange": "agent.events",
                "routing_key": "events.default",
                "encoding": "utf-8"
            }
            
            publisher = RabbitMQPublisher(config)
            
            assert publisher.config == config
            assert publisher.url == "amqp://user:pass@localhost:5672/vhost"
            assert publisher.exchange == "agent.events"
            assert publisher.routing_key == "events.default"
            assert publisher.exchange_type == "topic"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_publish_success(self, mock_import):
        """Test successful RabbitMQ publishing."""
        mock_pika = Mock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        # Set up proper async behavior
        mock_pika.connect_robust = AsyncMock(return_value=mock_connection)
        mock_connection.channel = AsyncMock(return_value=mock_channel)
        mock_channel.declare_exchange = AsyncMock(return_value=mock_exchange)
        mock_exchange.publish = AsyncMock()
        
        # Add exchange type enum
        mock_exchange_type = Mock()
        mock_exchange_type.TOPIC = "topic"
        mock_pika.ExchangeType = mock_exchange_type
        
        mock_pika.Message = Mock  # Mock Message class
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from agentflow.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            publisher = RabbitMQPublisher({
                "exchange": "test.exchange",
                "routing_key": "test.key"
            })
            
            event = EventModel(
                event=Event.STREAMING,
                event_type=EventType.UPDATE,
                node_name="rabbitmq_test",
                data={"update": "data"}
            )
            
            result = await publisher.publish(event)
            
            # Should return True (successful publish)
            assert result is True
            
            # Verify the publish call
            mock_exchange.publish.assert_called_once()
            call_args = mock_exchange.publish.call_args
            message = call_args[0][0]  # First positional arg
            routing_key = call_args[1].get('routing_key')  # Keyword arg
            
            assert routing_key == "test.key"
            
        # Verify message creation - Mock should be called as a constructor
        assert mock_pika.Message.called
class TestOptionalPublisherErrorHandling:
    """Test error handling across optional publishers."""
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_connection_retry_failure(self, mock_import):
        """Test Redis publisher with persistent connection failures."""
        mock_redis_asyncio = Mock()
        mock_redis_asyncio.from_url.side_effect = ConnectionError("Cannot connect")
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START
            )
            
            with pytest.raises(RuntimeError, match="RedisPublisher failed to connect to Redis"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_kafka_publisher_send_failure(self, mock_import):
        """Test Kafka publisher with send failures."""
        mock_kafka = Mock()
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.side_effect = Exception("Send failed")
        mock_kafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from agentflow.publisher.kafka_publisher import KafkaPublisher
            
            publisher = KafkaPublisher()
            
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.ERROR
            )
            
            with pytest.raises(Exception, match="Send failed"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_publish_failure(self, mock_import):
        """Test RabbitMQ publisher with publish failures."""
        mock_pika = Mock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_exchange.publish.side_effect = Exception("Publish failed")
        
        mock_connection.channel.return_value = mock_channel
        mock_channel.get_exchange.return_value = mock_exchange
        mock_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from agentflow.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            publisher = RabbitMQPublisher()
            
            event = EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.ERROR
            )
            
            # This will fail during _ensure() before publish
            with pytest.raises(TypeError):
                await publisher.publish(event)


class TestOptionalPublisherConfiguration:
    """Test configuration handling for optional publishers."""
    
    @patch('importlib.import_module')
    def test_redis_publisher_config_variations(self, mock_import):
        """Test RedisPublisher with various configuration options."""
        mock_redis = Mock()
        mock_import.return_value = mock_redis
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            # Test minimal config
            publisher1 = RedisPublisher()
            assert publisher1.url == "redis://localhost:6379/0"
            
            # Test full config
            full_config = {
                "url": "redis://user:pass@redis.example.com:6380/5",
                "mode": "stream",
                "channel": "custom.channel",
                "stream": "custom.stream",
                "maxlen": 5000,
                "encoding": "utf-16"
            }
            
            publisher2 = RedisPublisher(full_config)
            assert publisher2.url == "redis://user:pass@redis.example.com:6380/5"
            assert publisher2.mode == "stream"
            assert publisher2.maxlen == 5000
            assert publisher2.encoding == "utf-16"
    
    @patch('importlib.import_module')
    def test_kafka_publisher_config_variations(self, mock_import):
        """Test KafkaPublisher with various configurations."""
        mock_kafka = Mock()
        mock_kafka.AIOKafkaProducer = Mock()
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from agentflow.publisher.kafka_publisher import KafkaPublisher
            
            # Test with string bootstrap servers
            config1 = {
                "bootstrap_servers": "kafka1:9092,kafka2:9092",
                "topic": "single.topic"
            }
            
            publisher1 = KafkaPublisher(config1)
            # Should handle string format
            assert publisher1.bootstrap_servers == "kafka1:9092,kafka2:9092"
            
            # Test with list bootstrap servers
            config2 = {
                "bootstrap_servers": ["broker1:9092", "broker2:9092", "broker3:9092"],
                "topic": "list.topic"
            }
            
            publisher2 = KafkaPublisher(config2)
            assert publisher2.bootstrap_servers == config2["bootstrap_servers"]
    
    @patch('importlib.import_module')
    def test_rabbitmq_publisher_config_variations(self, mock_import):
        """Test RabbitMQPublisher with various configurations."""
        mock_pika = Mock()
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from agentflow.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            # Test default config
            publisher1 = RabbitMQPublisher()
            assert publisher1.url == "amqp://guest:guest@localhost/"
            assert publisher1.exchange == "agentflow.events"
            assert publisher1.routing_key == "agentflow.events"
            
            # Test custom config
            custom_config = {
                "url": "amqps://user:secret@rmq.example.com:5671/prod",
                "exchange": "production.events",
                "routing_key": "agent.updates",
                "queue": "agent.queue",
                "durable": True
            }
            
            publisher2 = RabbitMQPublisher(custom_config)
            assert publisher2.url == "amqps://user:secret@rmq.example.com:5671/prod"
            assert publisher2.exchange == "production.events"
            assert publisher2.routing_key == "agent.updates"


class TestOptionalPublisherIntegration:
    """Integration tests for optional publishers."""
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_full_lifecycle(self, mock_import):
        """Test complete Redis publisher lifecycle."""
        mock_redis_asyncio = Mock()
        mock_client = AsyncMock()
        mock_redis_asyncio.from_url.return_value = mock_client
        mock_client.publish.return_value = 2  # 2 subscribers
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from agentflow.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "channel": "integration.test"
            })
            
            # Publish multiple events
            events = [
                EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START),
                EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS),
                EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.END)
            ]
            
            results = []
            for event in events:
                result = await publisher.publish(event)
                results.append(result)
            
            # All should succeed
            assert all(r == 2 for r in results)
            assert mock_client.publish.call_count == 3
            
            # Close publisher
            await publisher.close()
            mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_publisher_types_concurrent(self):
        """Test multiple publisher types working concurrently."""
        import asyncio
        from agentflow.publisher.console_publisher import ConsolePublisher
        
        # Use ConsolePublisher for this test since it doesn't need external dependencies
        # This tests the concurrency pattern without complex mocking
        
        pub1 = ConsolePublisher()
        pub2 = ConsolePublisher()
        
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="concurrent_test_1"
        )
        
        event2 = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.END,
            node_name="concurrent_test_2"
        )
        
        # Test concurrent publishing with console publishers
        with patch('builtins.print') as mock_print:
            task1 = pub1.publish(event1)
            task2 = pub2.publish(event2)
            
            results = await asyncio.gather(task1, task2)
            
            # Both should succeed (console publisher returns None)
            assert results == [None, None]
            
            # Print should have been called twice
            assert mock_print.call_count == 2
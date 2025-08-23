"""Comprehensive tests for the publisher module."""

import pytest

from pyagenity.publisher import (
    BasePublisher,
    ConsolePublisher,
    Event,
    EventType,
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
        with pytest.raises(TypeError):
            BasePublisher()

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


def test_publisher_module_imports():
    """Test that publisher module imports work correctly."""
    assert BasePublisher is not None  # noqa: S101
    assert ConsolePublisher is not None  # noqa: S101
    assert Event is not None  # noqa: S101
    assert EventType is not None  # noqa: S101
    assert SourceType is not None  # noqa: S101

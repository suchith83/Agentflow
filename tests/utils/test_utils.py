"""Comprehensive tests for the utils module."""

import asyncio
import logging
import sys
from io import StringIO

import pytest

from pyagenity.utils import (
    END,
    START,
    CallbackContext,
    CallbackManager,
    Command,
    InvocationType,
    Message,
    add_messages,
    convert_messages,
    replace_messages,
    replace_value,
)
from pyagenity.utils.reducers import append_items
from pyagenity.utils.background_task_manager import BackgroundTaskManager
from pyagenity.utils.id_generator import (
    AsyncIDGenerator,
    BigIntIDGenerator,
    DefaultIDGenerator,
    HexIDGenerator,
    IDType,
    IntIDGenerator,
    ShortIDGenerator,
    TimestampIDGenerator,
    UUIDGenerator,
)
from pyagenity.utils.logging import configure_logging
from pyagenity.utils.message import TextBlock, TokenUsages, ToolResultBlock, generate_id


class TestMessage:
    """Test the Message class."""

    def test_message_text_message(self):
        """Test creating a message from text."""
        msg = Message.text_message("Hello world")
        assert msg.content[0].text == "Hello world"  # noqa: S101
        assert msg.role == "user"  # noqa: S101

    def test_message_text_message_with_role(self):
        """Test creating a message with specific role."""
        msg = Message.text_message("Hello", role="assistant")
        assert msg.content[0].text == "Hello"  # noqa: S101
        assert msg.role == "assistant"  # noqa: S101

    def test_message_model_dump(self):
        """Test converting message to dict using model_dump."""
        msg = Message.text_message("Test")
        msg_dict = msg.model_dump()
        assert isinstance(msg_dict, dict)  # noqa: S101
        assert "content" in msg_dict  # noqa: S101

    def test_message_copy(self):
        """Test copying a message."""
        msg = Message.text_message("Original")
        copied = msg.copy()
        assert copied.content == msg.content  # noqa: S101
        assert copied.role == msg.role  # noqa: S101

    def test_message_with_tools_calls(self):
        """Test message with tools_calls."""
        msg = Message(
            role="assistant",
            content=[TextBlock(text="Hello")],
            tools_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )
        assert msg.role == "assistant"  # noqa: S101
        assert msg.tools_calls is not None  # noqa: S101

    def test_message_from_dict_with_usages(self):
        """Test from_dict with usages parsing."""
        msg = Message(
            role="user",
            content=[],
            usages=TokenUsages(
                completion_tokens=10,
                prompt_tokens=20,
                total_tokens=30,
                reasoning_tokens=5,
            ),
        )
        assert msg.usages is not None  # noqa: S101
        assert msg.usages.completion_tokens == 10  # noqa: S101
        assert msg.usages.prompt_tokens == 20  # noqa: S101
        assert msg.usages.total_tokens == 30  # noqa: S101
        assert msg.usages.reasoning_tokens == 5  # noqa: S101


class TestGenerateId:
    """Test the generate_id function."""

    def test_generate_id_with_default_id_string(self):
        """Test generate_id with default_id matching string type."""
        from injectq import InjectQ

        iq = InjectQ.get_instance()
        iq.bind("generated_id_type", "string")

        try:
            result = generate_id("default_string")
            assert result == "default_string"  # noqa: S101
        finally:
            pass


class TestConstants:
    """Test the constants."""

    def test_start_constant(self):
        """Test START constant."""
        assert START is not None  # noqa: S101

    def test_end_constant(self):
        """Test END constant."""
        assert END is not None  # noqa: S101


class TestCommand:
    """Test the Command class."""

    def test_command_creation(self):
        """Test creating a Command with no args."""
        cmd = Command()
        assert cmd is not None  # noqa: S101

    def test_command_creation_with_goto(self):
        """Test creating a Command with goto."""
        cmd = Command(goto="next_node")
        assert cmd.goto == "next_node"  # noqa: S101


class TestCallbackManager:
    """Test the CallbackManager class."""

    def test_callback_manager_creation(self):
        """Test creating a CallbackManager."""
        manager = CallbackManager()
        assert manager is not None  # noqa: S101

    def test_callback_manager_methods(self):
        """Test CallbackManager has expected methods."""
        manager = CallbackManager()
        expected_methods = [
            "register_before_invoke",
            "register_after_invoke",
            "register_on_error",
            "execute_before_invoke",
            "execute_after_invoke",
            "execute_on_error",
            "clear_callbacks",
            "get_callback_counts",
        ]
        for method in expected_methods:
            assert hasattr(manager, method)  # noqa: S101
            assert callable(getattr(manager, method))  # noqa: S101

    @pytest.mark.asyncio
    async def test_callback_manager_execute_before_invoke(self):
        """Test executing before_invoke callbacks."""
        manager = CallbackManager()
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_func",
            metadata={},
        )

        async def callback(ctx, data):
            return data + "_modified"

        manager.register_before_invoke(InvocationType.AI, callback)
        result = await manager.execute_before_invoke(context, "input")
        assert result == "input_modified"  # noqa: S101

    @pytest.mark.asyncio
    async def test_callback_manager_execute_after_invoke(self):
        """Test executing after_invoke callbacks."""
        manager = CallbackManager()
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_func",
            metadata={},
        )

        async def callback(ctx, input_data, output_data):
            return output_data + "_processed"

        manager.register_after_invoke(InvocationType.AI, callback)
        result = await manager.execute_after_invoke(context, "input", "output")
        assert result == "output_processed"  # noqa: S101

    @pytest.mark.asyncio
    async def test_callback_manager_execute_on_error(self):
        """Test executing on_error callbacks."""
        manager = CallbackManager()
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_func",
            metadata={},
        )

        async def callback(ctx, input_data, error):
            return Message.text_message("recovery")

        manager.register_on_error(InvocationType.AI, callback)
        result = await manager.execute_on_error(context, "input", Exception("test"))
        assert isinstance(result, Message)  # noqa: S101

    def test_callback_manager_clear_callbacks(self):
        """Test clearing callbacks."""
        manager = CallbackManager()

        def callback(ctx, data):
            return data

        manager.register_before_invoke(InvocationType.AI, callback)
        counts = manager.get_callback_counts()
        assert counts["ai"]["before_invoke"] == 1  # noqa: S101

        manager.clear_callbacks(InvocationType.AI)
        counts = manager.get_callback_counts()
        assert counts["ai"]["before_invoke"] == 0  # noqa: S101

    def test_callback_manager_get_callback_counts(self):
        """Test getting callback counts."""
        manager = CallbackManager()
        counts = manager.get_callback_counts()
        assert isinstance(counts, dict)  # noqa: S101
        assert "ai" in counts  # noqa: S101
        assert "before_invoke" in counts["ai"]  # noqa: S101

    @pytest.mark.asyncio
    async def test_callback_manager_execute_before_invoke_sync_callable(self):
        """Test executing before_invoke with sync callable."""
        manager = CallbackManager()
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_func",
            metadata={},
        )

        def sync_callback(ctx, data):
            return data + "_sync_modified"

        manager.register_before_invoke(InvocationType.AI, sync_callback)
        result = await manager.execute_before_invoke(context, "input")
        assert result == "input_sync_modified"  # noqa: S101

    def test_callback_manager_clear_callbacks_all(self):
        """Test clearing all callbacks."""
        manager = CallbackManager()

        def callback(ctx, data):
            return data

        manager.register_before_invoke(InvocationType.AI, callback)
        manager.clear_callbacks()  # Clear all
        counts = manager.get_callback_counts()
        assert counts["ai"]["before_invoke"] == 0  # noqa: S101


class TestCallbackContext:
    """Test the CallbackContext class."""

    def test_callback_context_creation(self):
        """Test creating a CallbackContext."""
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_function",
            metadata={},
        )
        assert context.node_name == "test_node"  # noqa: S101
        assert context.invocation_type == InvocationType.AI  # noqa: S101


class TestInvocationType:
    """Test the InvocationType enum."""

    def test_invocation_type_values(self):
        """Test InvocationType enum values."""
        assert InvocationType.AI  # noqa: S101
        assert InvocationType.TOOL  # noqa: S101
        assert InvocationType.MCP  # noqa: S101


class TestReducers:
    """Comprehensive tests for reducer functions."""

    def test_add_messages_basic(self):
        """Test add_messages with basic functionality."""
        messages1 = [Message.text_message("Hello")]
        messages2 = [Message.text_message("World")]

        result = add_messages(messages1, messages2)
        assert isinstance(result, list)  # noqa: S101
        assert len(result) == 2  # noqa: S101

    def test_add_messages_empty_lists(self):
        """Test add_messages with empty lists."""
        # Empty left
        result = add_messages([], [Message.text_message("Hello")])
        assert len(result) == 1  # noqa: S101

        # Empty right
        messages = [Message.text_message("Hello")]
        result = add_messages(messages, [])
        assert len(result) == 1  # noqa: S101
        assert result == messages  # noqa: S101

        # Both empty
        result = add_messages([], [])
        assert len(result) == 0  # noqa: S101

    def test_add_messages_avoid_duplicates(self):
        """Test add_messages avoids duplicates by message_id."""
        msg1 = Message.text_message("Hello")
        msg2 = Message.text_message("World")
        
        # Same message in both lists should not be duplicated
        result = add_messages([msg1, msg2], [msg2, Message.text_message("New")])
        assert len(result) == 3  # noqa: S101
        
        # Verify no duplicate message_ids
        message_ids = [msg.message_id for msg in result]
        assert len(message_ids) == len(set(message_ids))  # noqa: S101

    def test_add_messages_preserves_order(self):
        """Test add_messages preserves order from left list first."""
        msg1 = Message.text_message("First")
        msg2 = Message.text_message("Second") 
        msg3 = Message.text_message("Third")

        result = add_messages([msg1], [msg2, msg3])
        assert result[0].message_id == msg1.message_id  # noqa: S101
        assert result[1].message_id == msg2.message_id  # noqa: S101
        assert result[2].message_id == msg3.message_id  # noqa: S101

    def test_replace_messages_basic(self):
        """Test replace_messages basic functionality."""
        messages1 = [Message.text_message("Hello")]
        messages2 = [Message.text_message("World")]

        result = replace_messages(messages1, messages2)
        assert result == messages2  # noqa: S101
        assert result is not messages1  # noqa: S101

    def test_replace_messages_empty_lists(self):
        """Test replace_messages with empty lists."""
        messages = [Message.text_message("Hello")]
        
        # Replace with empty list
        result = replace_messages(messages, [])
        assert len(result) == 0  # noqa: S101

        # Replace empty list with messages
        result = replace_messages([], messages)
        assert result == messages  # noqa: S101

    def test_replace_messages_ignores_left(self):
        """Test that replace_messages completely ignores left argument."""
        left = [Message.text_message("Ignored")]
        right = [Message.text_message("Used")]
        
        result = replace_messages(left, right)
        assert result == right  # noqa: S101
        assert len(result) == 1  # noqa: S101

    def test_append_items_basic(self):
        """Test append_items with basic functionality."""
        class TestItem:
            def __init__(self, id_val, data):
                self.id = id_val
                self.data = data
            
            def __eq__(self, other):
                return self.id == other.id and self.data == other.data

        item1 = TestItem("1", "data1")
        item2 = TestItem("2", "data2")
        
        result = append_items([item1], [item2])
        assert len(result) == 2  # noqa: S101
        assert result[0] == item1  # noqa: S101
        assert result[1] == item2  # noqa: S101

    def test_append_items_avoid_duplicates(self):
        """Test append_items avoids duplicates by id."""
        class TestItem:
            def __init__(self, id_val, data):
                self.id = id_val
                self.data = data

        item1 = TestItem("1", "original")
        item1_duplicate = TestItem("1", "duplicate")
        item2 = TestItem("2", "unique")
        
        result = append_items([item1], [item1_duplicate, item2])
        assert len(result) == 2  # noqa: S101
        # Should keep the original, not the duplicate
        assert result[0].data == "original"  # noqa: S101
        assert result[1].data == "unique"  # noqa: S101

    def test_append_items_empty_lists(self):
        """Test append_items with empty lists."""
        class TestItem:
            def __init__(self, id_val):
                self.id = id_val

        item = TestItem("1")
        
        # Empty left
        result = append_items([], [item])
        assert len(result) == 1  # noqa: S101

        # Empty right
        result = append_items([item], [])
        assert len(result) == 1  # noqa: S101

        # Both empty
        result = append_items([], [])
        assert len(result) == 0  # noqa: S101

    def test_replace_value_basic(self):
        """Test replace_value with basic types."""
        assert replace_value("old", "new") == "new"  # noqa: S101
        assert replace_value(1, 2) == 2  # noqa: S101
        assert replace_value([], ["new"]) == ["new"]  # noqa: S101

    def test_replace_value_ignores_left(self):
        """Test that replace_value completely ignores left argument."""
        assert replace_value("ignored", "used") == "used"  # noqa: S101
        assert replace_value(100, "string") == "string"  # noqa: S101
        assert replace_value(None, False) is False  # noqa: S101

    def test_replace_value_none_values(self):
        """Test replace_value with None values."""
        assert replace_value("something", None) is None  # noqa: S101
        assert replace_value(None, "something") == "something"  # noqa: S101
        assert replace_value(None, None) is None  # noqa: S101


class TestConverter:
    """Test the converter functions."""

    def test_convert_messages_basic(self):
        """Test convert_messages with basic system prompts."""
        system_prompts = [{"role": "system", "content": "You are a helpful assistant."}]
        result = convert_messages(system_prompts)
        assert isinstance(result, list)  # noqa: S101
        assert len(result) == 1  # noqa: S101
        assert result[0]["role"] == "system"  # noqa: S101

    def test_convert_messages_with_state_context_summary(self):
        """Test convert_messages with state having context summary."""
        from pyagenity.state import AgentState

        system_prompts = [{"role": "system", "content": "Test"}]
        state = AgentState()
        state.context_summary = "Summary of context"
        result = convert_messages(system_prompts, state)
        assert len(result) == 2  # noqa: S101
        assert result[1]["role"] == "assistant"  # noqa: S101
        assert result[1]["content"] == "Summary of context"  # noqa: S101

    def test_convert_messages_with_state_context(self):
        """Test convert_messages with state having context messages."""
        from pyagenity.state import AgentState

        system_prompts = [{"role": "system", "content": "Test"}]
        state = AgentState()
        state.context = [Message.text_message("Hello", "user")]
        result = convert_messages(system_prompts, state)
        assert len(result) == 2  # noqa: S101
        assert result[1]["role"] == "user"  # noqa: S101
        assert result[1]["content"] == "Hello"  # noqa: S101

    def test_convert_messages_with_extra_messages(self):
        """Test convert_messages with extra messages."""
        system_prompts = [{"role": "system", "content": "Test"}]
        extra_messages = [Message.text_message("Extra", "assistant")]
        result = convert_messages(system_prompts, extra_messages=extra_messages)
        assert len(result) == 2  # noqa: S101
        assert result[1]["role"] == "assistant"  # noqa: S101

    def test_convert_messages_none_system_prompts(self):
        """Test convert_messages with None system prompts raises error."""
        import pytest

        with pytest.raises(ValueError, match="System prompts cannot be None"):
            convert_messages(None)  # type: ignore

    def test_convert_messages_tool_message(self):
        """Test converting tool messages."""
        from pyagenity.state import AgentState

        system_prompts = [{"role": "system", "content": "Test"}]
        state = AgentState()
        tool_msg = Message(
            role="tool",
            content=[
                ToolResultBlock(
                    call_id="call_123",
                    output="Tool output",
                    is_error=False,
                )
            ],
            tools_calls=[],
            message_id="call_123",
        )
        state.context = [tool_msg]
        result = convert_messages(system_prompts, state)
        assert len(result) == 2  # noqa: S101
        assert result[1]["role"] == "tool"  # noqa: S101
        assert result[1]["tool_call_id"] == "call_123"  # noqa: S101


class TestConvenienceFunctions:
    """Test the convenience functions for global callback manager."""

    def test_register_before_invoke_convenience(self):
        """Test register_before_invoke convenience function."""
        from pyagenity.utils.callbacks import register_before_invoke

        def callback(ctx, data):
            return data

        # Should not raise
        register_before_invoke(InvocationType.AI, callback)

    def test_register_after_invoke_convenience(self):
        """Test register_after_invoke convenience function."""
        from pyagenity.utils.callbacks import register_after_invoke

        def callback(ctx, input_data, output_data):
            return output_data

        # Should not raise
        register_after_invoke(InvocationType.AI, callback)

    def test_register_on_error_convenience(self):
        """Test register_on_error convenience function."""
        from pyagenity.utils.callbacks import register_on_error

        def callback(ctx, input_data, error):
            return None

        # Should not raise
        register_on_error(InvocationType.AI, callback)


class TestIDGenerators:
    """Test the ID generator classes."""

    def test_uuid_generator(self):
        """Test UUIDGenerator."""
        generator = UUIDGenerator()
        assert generator.id_type == IDType.STRING  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, str)  # noqa: S101
        assert len(id_val) == 36  # noqa: S101

    def test_bigint_id_generator(self):
        """Test BigIntIDGenerator."""
        generator = BigIntIDGenerator()
        assert generator.id_type == IDType.BIGINT  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, int)  # noqa: S101
        assert id_val > 0  # noqa: S101

    def test_default_id_generator(self):
        """Test DefaultIDGenerator."""
        generator = DefaultIDGenerator()
        assert generator.id_type == IDType.STRING  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, str)  # noqa: S101
        assert id_val == ""  # noqa: S101

    def test_int_id_generator(self):
        """Test IntIDGenerator."""
        generator = IntIDGenerator()
        assert generator.id_type == IDType.INTEGER  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, int)  # noqa: S101
        assert 0 <= id_val <= 2**32 - 1  # noqa: S101

    def test_hex_id_generator(self):
        """Test HexIDGenerator."""
        generator = HexIDGenerator()
        assert generator.id_type == IDType.STRING  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, str)  # noqa: S101
        assert len(id_val) == 32  # noqa: S101

    def test_timestamp_id_generator(self):
        """Test TimestampIDGenerator."""
        generator = TimestampIDGenerator()
        assert generator.id_type == IDType.INTEGER  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, int)  # noqa: S101
        assert id_val > 0  # noqa: S101

    def test_short_id_generator(self):
        """Test ShortIDGenerator."""
        generator = ShortIDGenerator()
        assert generator.id_type == IDType.STRING  # noqa: S101
        id_val = generator.generate()
        assert isinstance(id_val, str)  # noqa: S101
        assert len(id_val) == 8  # noqa: S101

    @pytest.mark.asyncio
    async def test_async_id_generator(self):
        """Test AsyncIDGenerator."""
        generator = AsyncIDGenerator()
        assert generator.id_type == IDType.STRING  # noqa: S101
        id_val = await generator.generate()
        assert isinstance(id_val, str)  # noqa: S101
        assert len(id_val) == 36  # noqa: S101


class TestBackgroundTaskManager:
    """Test the BackgroundTaskManager class."""

    @pytest.mark.asyncio
    async def test_create_task_and_wait(self):
        """Test creating a task and waiting for it."""
        manager = BackgroundTaskManager()

        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"

        manager.create_task(dummy_task())
        await manager.wait_for_all()
        assert len(manager._tasks) == 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_task_with_exception(self):
        """Test task that raises exception."""
        manager = BackgroundTaskManager()

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            manager.create_task(failing_task())
            await manager.wait_for_all()
            assert len(manager._tasks) == 0  # noqa: S101


def test_utils_module_imports():
    """Test that utils module imports work correctly."""
    # Basic smoke test - just ensure imports work
    assert Message is not None  # noqa: S101
    assert START is not None  # noqa: S101
    assert END is not None  # noqa: S101
    assert CallbackManager is not None  # noqa: S101
    assert InvocationType is not None  # noqa: S101


class TestLogging:
    """Test the logging configuration module."""

    def test_configure_logging_default(self):
        """Test configure_logging with default parameters."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure with defaults
        configure_logging()

        # Verify logger configuration
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[0].stream == sys.stdout
        assert not logger.propagate

        # Verify formatter
        formatter = logger.handlers[0].formatter
        assert formatter is not None
        # Test that formatter works
        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "test message", (), None)
        formatted = formatter.format(record)
        assert "[20" in formatted  # Should contain timestamp
        assert "INFO" in formatted
        assert "test" in formatted
        assert "test message" in formatted

    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom log level."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure with DEBUG level
        configure_logging(level=logging.DEBUG)

        # Verify logger configuration
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

    def test_configure_logging_custom_format(self):
        """Test configure_logging with custom format string."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure with custom format
        custom_format = "%(levelname)s: %(message)s"
        configure_logging(format_string=custom_format)

        # Verify formatter
        formatter = logger.handlers[0].formatter
        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "test message", (), None)
        formatted = formatter.format(record)
        assert formatted == "INFO: test message"

    def test_configure_logging_custom_handler(self):
        """Test configure_logging with custom handler."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Create custom handler
        string_stream = StringIO()
        custom_handler = logging.StreamHandler(string_stream)

        # Configure with custom handler
        configure_logging(handler=custom_handler)

        # Verify handler is used
        assert len(logger.handlers) == 1
        assert logger.handlers[0] is custom_handler
        assert logger.handlers[0].stream == string_stream

    def test_configure_logging_no_duplicate_handlers(self):
        """Test that configure_logging doesn't add duplicate handlers."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure logging twice
        configure_logging()
        initial_handler_count = len(logger.handlers)

        configure_logging()
        final_handler_count = len(logger.handlers)

        # Should not add duplicate handlers
        assert final_handler_count == initial_handler_count

    def test_configure_logging_preserves_existing_handlers(self):
        """Test that configure_logging replaces existing handlers."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Add a custom handler manually
        custom_handler = logging.StreamHandler(StringIO())
        logger.addHandler(custom_handler)

        # Configure logging - replaces existing handler
        configure_logging()

        # Should have only one handler (configure_logging replaces existing)
        assert len(logger.handlers) == 1

    def test_configure_logging_with_all_custom_params(self):
        """Test configure_logging with all custom parameters."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure with all custom parameters
        string_stream = StringIO()
        custom_handler = logging.StreamHandler(string_stream)
        custom_format = "%(name)s - %(levelname)s: %(message)s"

        configure_logging(
            level=logging.WARNING, format_string=custom_format, handler=custom_handler
        )

        # Verify all configurations
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1
        assert logger.handlers[0] is custom_handler
        assert not logger.propagate

        # Test custom formatter
        formatter = logger.handlers[0].formatter
        record = logging.LogRecord(
            "test.logger", logging.WARNING, "test.py", 1, "warning message", (), None
        )
        formatted = formatter.format(record)
        assert formatted == "test.logger - WARNING: warning message"

    def test_default_configuration_on_import(self):
        """Test that default configuration is applied on module import."""
        # Get the logger
        logger = logging.getLogger("pyagenity")

        # Should have at least one handler (configured on import)
        assert len(logger.handlers) >= 1

        # Should not propagate
        assert not logger.propagate

    def test_logger_hierarchy(self):
        """Test that module-specific loggers work correctly."""
        # Clear pyagenity logger
        pyagenity_logger = logging.getLogger("pyagenity")
        pyagenity_logger.handlers.clear()

        # Configure pyagenity logging
        configure_logging()

        # Create module-specific logger
        module_logger = logging.getLogger("pyagenity.test_module")

        # Module logger should inherit configuration
        assert module_logger.level == logging.NOTSET  # Inherits from parent
        assert module_logger.propagate  # Should propagate to pyagenity logger

        # Test logging through module logger
        import io

        # Create a string handler to capture log output
        log_output = io.StringIO()
        handler = logging.StreamHandler(log_output)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)
        module_logger.setLevel(logging.INFO)

        try:
            module_logger.info("Test message")

            # Verify that the message was logged
            output = log_output.getvalue()
            assert "Test message" in output
            assert "pyagenity.test_module" in output
        finally:
            # Clean up the handler
            module_logger.removeHandler(handler)

    def test_logging_output_capture(self):
        """Test that logging actually outputs to the configured stream."""
        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Configure with string stream
        string_stream = StringIO()
        handler = logging.StreamHandler(string_stream)
        configure_logging(handler=handler)

        # Log a message
        logger.info("Test log message")

        # Verify output
        output = string_stream.getvalue()
        assert "Test log message" in output
        assert "INFO" in output
        assert "pyagenity" in output

    def test_configure_logging_with_file_handler(self):
        """Test configure_logging with a file handler."""
        import os
        import tempfile

        # Clear any existing handlers
        logger = logging.getLogger("pyagenity")
        logger.handlers.clear()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Create file handler
            file_handler = logging.FileHandler(temp_filename)

            # Configure logging
            configure_logging(handler=file_handler)

            # Log a message
            logger.info("Test file log message")

            # Read the file
            with open(temp_filename) as f:
                content = f.read()

            # Verify content
            assert "Test file log message" in content
            assert "INFO" in content

        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

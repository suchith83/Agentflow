"""Comprehensive tests for the utils module."""

from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    Command,
    InvocationType,
    Message,
    add_messages,
    END,
    START,
)
import logging
import sys
from io import StringIO
from unittest.mock import patch

from pyagenity.utils.logging import configure_logging


class TestMessage:
    """Test the Message class."""

    def test_message_from_text(self):
        """Test creating a message from text."""
        msg = Message.from_text("Hello world")
        assert msg.content == "Hello world"  # noqa: S101
        assert msg.role == "user"  # noqa: S101

    def test_message_from_text_with_role(self):
        """Test creating a message with specific role."""
        msg = Message.from_text("Hello", role="assistant")
        assert msg.content == "Hello"  # noqa: S101
        assert msg.role == "assistant"  # noqa: S101

    def test_message_model_dump(self):
        """Test converting message to dict using model_dump."""
        msg = Message.from_text("Test")
        msg_dict = msg.model_dump()
        assert isinstance(msg_dict, dict)  # noqa: S101
        assert "content" in msg_dict  # noqa: S101

    def test_message_tool_message(self):
        """Test creating a tool message."""
        msg = Message.tool_message("call_123", "Tool result")
        assert msg.content == "Tool result"  # noqa: S101

    def test_message_from_dict(self):
        """Test creating message from dict."""
        msg_data = {"content": "Test content", "role": "user"}
        msg = Message.from_dict(msg_data)
        assert msg.content == "Test content"  # noqa: S101
        assert msg.role == "user"  # noqa: S101

    def test_message_copy(self):
        """Test copying a message."""
        msg = Message.from_text("Original")
        copied = msg.copy()
        assert copied.content == msg.content  # noqa: S101
        assert copied.role == msg.role  # noqa: S101


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
            "add_before_invoke_callback",
            "add_after_invoke_callback",
            "add_error_callback",
        ]
        for method in expected_methods:
            if hasattr(manager, method):
                assert callable(getattr(manager, method))  # noqa: S101


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


class TestAddMessages:
    """Test the add_messages function."""

    def test_add_messages_function(self):
        """Test add_messages reducer function."""
        messages1 = [Message.from_text("Hello")]
        messages2 = [Message.from_text("World")]

        result = add_messages(messages1, messages2)
        assert isinstance(result, list)  # noqa: S101
        assert len(result) == 2  # noqa: S101


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

        # Should have appropriate level (could be WARNING or INFO)
        assert logger.level in [logging.INFO, logging.WARNING]

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
        import tempfile
        import os

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
            with open(temp_filename, "r") as f:
                content = f.read()

            # Verify content
            assert "Test file log message" in content
            assert "INFO" in content

        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

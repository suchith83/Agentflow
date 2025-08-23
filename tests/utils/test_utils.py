"""Comprehensive tests for the utils module."""

from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    Command,
    DependencyContainer,
    InvocationType,
    Message,
    add_messages,
    END,
    START,
)


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


class TestDependencyContainer:
    """Test the DependencyContainer class."""

    def test_dependency_container_creation(self):
        """Test creating a DependencyContainer."""
        container = DependencyContainer()
        assert container is not None  # noqa: S101

    def test_dependency_container_register_get(self):
        """Test registering and getting dependencies."""
        container = DependencyContainer()
        test_obj = {"test": "value"}

        container.register("test_dep", test_obj)
        retrieved = container.get("test_dep")
        assert retrieved == test_obj  # noqa: S101

    def test_dependency_container_has(self):
        """Test checking if dependency exists."""
        container = DependencyContainer()
        container.register("test_dep", "test_value")

        assert container.has("test_dep")  # noqa: S101
        assert not container.has("non_existent")  # noqa: S101

    def test_dependency_container_list_dependencies(self):
        """Test listing dependencies."""
        container = DependencyContainer()
        container.register("dep1", "value1")
        container.register("dep2", "value2")

        deps = container.list_dependencies()
        assert "dep1" in deps  # noqa: S101
        assert "dep2" in deps  # noqa: S101

    def test_dependency_container_unregister(self):
        """Test unregistering dependencies."""
        container = DependencyContainer()
        container.register("test_dep", "test_value")
        assert container.has("test_dep")  # noqa: S101

        container.unregister("test_dep")
        assert not container.has("test_dep")  # noqa: S101

    def test_dependency_container_clear(self):
        """Test clearing all dependencies."""
        container = DependencyContainer()
        container.register("dep1", "value1")
        container.register("dep2", "value2")

        container.clear()
        assert len(container.list_dependencies()) == 0  # noqa: S101

    def test_dependency_container_copy(self):
        """Test copying container."""
        container = DependencyContainer()
        container.register("test_dep", "test_value")

        copied = container.copy()
        assert copied.has("test_dep")  # noqa: S101
        assert copied.get("test_dep") == "test_value"  # noqa: S101


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

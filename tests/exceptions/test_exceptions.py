"""Comprehensive tests for the exceptions module."""

import pytest

from pyagenity.exceptions import (
    GraphError,
    GraphRecursionError,
    NodeError,
)


class TestGraphError:
    """Test the GraphError exception."""

    def test_graph_error_creation(self):
        """Test creating a GraphError."""
        error = GraphError("Test graph error")
        assert str(error) == "Test graph error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_graph_error_with_cause(self):
        """Test creating a GraphError with a cause."""
        cause = ValueError("Original error")
        try:
            raise GraphError("Graph failed") from cause
        except GraphError as error:
            assert str(error) == "Graph failed"  # noqa: S101
            assert error.__cause__ == cause  # noqa: S101

    def test_graph_error_inheritance(self):
        """Test that GraphError inherits from Exception."""
        error = GraphError("Test error")
        assert isinstance(error, Exception)  # noqa: S101

    def test_graph_error_raise(self):
        """Test raising a GraphError."""
        with pytest.raises(GraphError) as exc_info:
            raise GraphError("Test graph error")

        assert str(exc_info.value) == "Test graph error"  # noqa: S101


class TestNodeError:
    """Test the NodeError exception."""

    def test_node_error_creation(self):
        """Test creating a NodeError."""
        error = NodeError("Test node error")
        assert str(error) == "Test node error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_node_error_with_node_name(self):
        """Test creating a NodeError with node context."""
        error = NodeError("Node 'agent' failed: Invalid input")
        assert "agent" in str(error)  # noqa: S101
        assert "failed" in str(error)  # noqa: S101

    def test_node_error_inheritance(self):
        """Test that NodeError inherits from GraphError."""
        error = NodeError("Test error")
        assert isinstance(error, GraphError)  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_node_error_raise(self):
        """Test raising a NodeError."""
        with pytest.raises(NodeError) as exc_info:
            raise NodeError("Test node error")

        assert str(exc_info.value) == "Test node error"  # noqa: S101


class TestGraphRecursionError:
    """Test the GraphRecursionError exception."""

    def test_recursion_error_creation(self):
        """Test creating a GraphRecursionError."""
        error = GraphRecursionError("Test recursion error")
        assert str(error) == "Test recursion error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_recursion_error_with_depth(self):
        """Test creating a GraphRecursionError with depth information."""
        error = GraphRecursionError("Maximum recursion depth exceeded: 100")
        assert "recursion" in str(error)  # noqa: S101
        assert "100" in str(error)  # noqa: S101

    def test_recursion_error_inheritance(self):
        """Test that GraphRecursionError inherits from GraphError."""
        error = GraphRecursionError("Test error")
        assert isinstance(error, GraphError)  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_recursion_error_raise(self):
        """Test raising a GraphRecursionError."""
        with pytest.raises(GraphRecursionError) as exc_info:
            raise GraphRecursionError("Test recursion error")

        assert str(exc_info.value) == "Test recursion error"  # noqa: S101


class TestExceptionHierarchy:
    """Test exception hierarchy and relationships."""

    def test_all_exceptions_are_exceptions(self):
        """Test that all custom exceptions inherit from Exception."""
        errors = [
            GraphError("test"),
            NodeError("test"),
            GraphRecursionError("test"),
        ]

        for error in errors:
            assert isinstance(error, Exception)  # noqa: S101

    def test_node_error_is_graph_error(self):
        """Test that NodeError inherits from GraphError."""
        error = NodeError("test")
        assert isinstance(error, GraphError)  # noqa: S101

    def test_recursion_error_is_graph_error(self):
        """Test that GraphRecursionError inherits from GraphError."""
        error = GraphRecursionError("test")
        assert isinstance(error, GraphError)  # noqa: S101

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        original_error = ValueError("Original problem")

        try:
            raise NodeError("Node failed") from original_error
        except NodeError as node_error:
            assert node_error.__cause__ == original_error  # noqa: S101
            assert isinstance(node_error, NodeError)  # noqa: S101

    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        try:
            # First exception
            try:
                raise ValueError("First error")
            except ValueError:
                # Second exception that should preserve context
                raise GraphError("Graph error occurred")
        except GraphError as graph_error:
            assert graph_error.__context__ is not None  # noqa: S101
            assert isinstance(graph_error.__context__, ValueError)  # noqa: S101


def test_exceptions_module_imports():
    """Test that exceptions module imports work correctly."""
    assert GraphError is not None  # noqa: S101
    assert NodeError is not None  # noqa: S101
    assert GraphRecursionError is not None  # noqa: S101

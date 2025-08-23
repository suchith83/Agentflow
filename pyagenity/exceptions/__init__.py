"""
pyagenity.exceptions
====================

This package contains custom exception classes for graph operations in PyAgenity.

Available exceptions:
        - GraphError: Base exception for graph-related errors.
        - NodeError: Exception for node-specific errors.
        - GraphRecursionError: Exception for recursion limit errors in graphs.
"""

from .graph_error import GraphError
from .node_error import NodeError
from .recursion_error import GraphRecursionError


__all__ = ["GraphError", "GraphRecursionError", "NodeError"]

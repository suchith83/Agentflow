"""ToolNode package.

This package provides a modularized implementation of ToolNode. Public API:

- ToolNode
- HAS_FASTMCP, HAS_MCP

Backwards-compatible import path: ``from agentflow.graph.tool_node import ToolNode``
"""

from .base import ToolNode  # re-export
from .deps import HAS_FASTMCP, HAS_MCP  # re-export


__all__ = ["HAS_FASTMCP", "HAS_MCP", "ToolNode"]

"""Dependency flags and optional imports for ToolNode.

This isolates optional third-party imports so other modules can import
flags without triggering ImportError cascades.
"""

from __future__ import annotations


try:  # Optional fastmcp
    from fastmcp import Client  # type: ignore
    from fastmcp.client.client import CallToolResult  # type: ignore

    HAS_FASTMCP = True
except Exception:  # pragma: no cover - optional
    HAS_FASTMCP = False
    Client = None  # type: ignore
    CallToolResult = None  # type: ignore

try:  # Optional MCP protocol
    from mcp import Tool  # type: ignore
    from mcp.types import ContentBlock  # type: ignore

    HAS_MCP = True
except Exception:  # pragma: no cover - optional
    HAS_MCP = False
    Tool = None  # type: ignore
    ContentBlock = None  # type: ignore

__all__ = ["HAS_FASTMCP", "HAS_MCP", "CallToolResult", "Client", "ContentBlock", "Tool"]

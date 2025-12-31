"""Mock MCP client for testing.

This module provides a MockMCPClient class for testing MCP tool integrations
without requiring actual MCP servers or external connections.

Example:
    ```python
    from agentflow.testing import MockMCPClient
    from agentflow.graph.tool_node import ToolNode

    # Create mock MCP client
    mock_mcp = MockMCPClient()
    mock_mcp.add_tool(
        name="mcp_weather",
        description="Get weather for a city",
        parameters={"city": {"type": "string"}},
        handler=lambda city: f"Weather in {city}: Sunny",
    )

    # Use with ToolNode
    tool_node = ToolNode([], client=mock_mcp)
    tools = await tool_node.all_tools()
    # tools will include mcp_weather
    ```
"""

import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger("agentflow.testing")


class MockMCPClient:
    """Mock MCP client for testing ToolNode with MCP tools.

    Simulates an MCP client without requiring actual MCP servers.
    Allows registering mock tools with custom handlers for testing.

    Attributes:
        tools: Dictionary of registered mock tools
        calls: Dictionary tracking calls made to each tool

    Example:
        ```python
        mock_client = MockMCPClient()
        mock_client.add_tool(
            name="search",
            description="Search the web",
            parameters={"query": {"type": "string"}},
            handler=lambda query: f"Results for: {query}",
        )

        # Pass to ToolNode
        tools = ToolNode([], client=mock_client)
        ```
    """

    def __init__(self):
        """Initialize an empty mock MCP client."""
        self.tools: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        logger.debug("MockMCPClient initialized")

    def add_tool(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        handler: Callable | None = None,
    ) -> "MockMCPClient":
        """Register a mock MCP tool.

        Args:
            name: Name of the tool
            description: Tool description for LLM
            parameters: JSON schema for tool parameters
            handler: Function to call when tool is invoked

        Returns:
            Self for method chaining
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": parameters or {},
                "required": list((parameters or {}).keys()),
            },
            "handler": handler or (lambda **kwargs: f"Mock result for {name}"),
        }
        logger.debug("Registered mock MCP tool: %s", name)
        return self

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools (MCP protocol method).

        Returns:
            List of tool definitions in MCP format
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for tool in self.tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name (MCP protocol method).

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool doesn't exist
        """
        if name not in self.tools:
            raise ValueError(f"Unknown MCP tool: {name}")

        # Track the call
        if name not in self.calls:
            self.calls[name] = []
        self.calls[name].append({"arguments": arguments})

        # Execute handler
        handler = self.tools[name]["handler"]
        logger.debug("Calling mock MCP tool '%s' with args: %s", name, arguments)

        if callable(handler):
            # Support both sync and async handlers
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
        else:
            result = handler

        return result

    def was_called(self, name: str) -> bool:
        """Check if a tool was called at least once.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool was called
        """
        return name in self.calls and len(self.calls[name]) > 0

    def call_count(self, name: str) -> int:
        """Get number of times a tool was called.

        Args:
            name: Name of the tool

        Returns:
            Number of calls
        """
        return len(self.calls.get(name, []))

    def get_calls(self, name: str) -> list[dict[str, Any]]:
        """Get all calls made to a tool.

        Args:
            name: Name of the tool

        Returns:
            List of call records with 'arguments' key
        """
        return self.calls.get(name, [])

    def get_last_call(self, name: str) -> dict[str, Any] | None:
        """Get the most recent call to a tool.

        Args:
            name: Name of the tool

        Returns:
            Call record or None if never called
        """
        calls = self.get_calls(name)
        return calls[-1] if calls else None

    def assert_called(self, name: str) -> None:
        """Assert that a tool was called at least once.

        Args:
            name: Name of the tool

        Raises:
            AssertionError: If tool was never called
        """
        assert self.was_called(name), f"MCP tool '{name}' was never called"

    def assert_called_with(self, name: str, **expected_args: Any) -> None:
        """Assert that a tool was called with specific arguments.

        Checks the last call to the tool.

        Args:
            name: Name of the tool
            **expected_args: Expected arguments

        Raises:
            AssertionError: If tool not called or args don't match
        """
        last_call = self.get_last_call(name)
        assert last_call is not None, f"MCP tool '{name}' was never called"

        for key, expected in expected_args.items():
            actual = last_call["arguments"].get(key)
            assert actual == expected, (
                f"MCP tool '{name}' called with {key}={actual}, expected {key}={expected}"
            )

    def reset(self) -> None:
        """Clear call history (keeps tool registrations)."""
        self.calls.clear()
        logger.debug("MockMCPClient call history cleared")

    def clear(self) -> None:
        """Clear both tools and call history."""
        self.tools.clear()
        self.calls.clear()
        logger.debug("MockMCPClient completely cleared")


class MockComposioAdapter:
    """Mock Composio adapter for testing Composio tool integrations.

    Simulates a Composio adapter without requiring actual Composio connections.

    Example:
        ```python
        mock_composio = MockComposioAdapter()
        mock_composio.add_tool(
            slug="GITHUB_CREATE_ISSUE",
            description="Create a GitHub issue",
            parameters={"title": {"type": "string"}, "body": {"type": "string"}},
            handler=lambda title, body: {"issue_number": 123},
        )

        tools = ToolNode([], composio_adapter=mock_composio)
        ```
    """

    def __init__(self):
        """Initialize an empty mock Composio adapter."""
        self.tools: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        logger.debug("MockComposioAdapter initialized")

    def add_tool(
        self,
        slug: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        handler: Callable | None = None,
    ) -> "MockComposioAdapter":
        """Register a mock Composio tool.

        Args:
            slug: Tool slug (e.g., "GITHUB_CREATE_ISSUE")
            description: Tool description
            parameters: JSON schema for parameters
            handler: Function to call when tool is invoked

        Returns:
            Self for method chaining
        """
        self.tools[slug] = {
            "slug": slug,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
                "required": list((parameters or {}).keys()),
            },
            "handler": handler or (lambda **kwargs: {"result": f"Mock result for {slug}"}),
        }
        logger.debug("Registered mock Composio tool: %s", slug)
        return self

    def list_raw_tools_for_llm(self) -> list[dict[str, Any]]:
        """List tools in Composio format.

        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["slug"].lower(),
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.tools.values()
        ]

    def execute(
        self,
        slug: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a Composio tool.

        Args:
            slug: Tool slug to execute
            arguments: Tool arguments
            **kwargs: Additional execution context

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool doesn't exist
        """
        # Normalize slug (Composio uses lowercase)
        slug_upper = slug.upper()
        if slug_upper not in self.tools:
            raise ValueError(f"Unknown Composio tool: {slug}")

        # Track call
        if slug_upper not in self.calls:
            self.calls[slug_upper] = []
        self.calls[slug_upper].append({"arguments": arguments, "kwargs": kwargs})

        # Execute handler
        handler = self.tools[slug_upper]["handler"]
        logger.debug("Calling mock Composio tool '%s' with args: %s", slug, arguments)

        if callable(handler):
            return handler(**arguments)
        return handler

    def was_called(self, slug: str) -> bool:
        """Check if a tool was called."""
        return slug.upper() in self.calls and len(self.calls[slug.upper()]) > 0

    def call_count(self, slug: str) -> int:
        """Get number of times a tool was called."""
        return len(self.calls.get(slug.upper(), []))

    def get_calls(self, slug: str) -> list[dict[str, Any]]:
        """Get all calls made to a tool."""
        return self.calls.get(slug.upper(), [])

    def reset(self) -> None:
        """Clear call history."""
        self.calls.clear()

    def clear(self) -> None:
        """Clear tools and call history."""
        self.tools.clear()
        self.calls.clear()


class MockLangChainAdapter:
    """Mock LangChain adapter for testing LangChain tool integrations.

    Simulates a LangChain tool adapter for testing.

    Example:
        ```python
        mock_langchain = MockLangChainAdapter()
        mock_langchain.add_tool(
            name="calculator",
            description="Perform calculations",
            handler=lambda expression: eval(expression),
        )

        tools = ToolNode([], langchain_adapter=mock_langchain)
        ```
    """

    def __init__(self):
        """Initialize an empty mock LangChain adapter."""
        self.tools: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        logger.debug("MockLangChainAdapter initialized")

    def add_tool(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        handler: Callable | None = None,
    ) -> "MockLangChainAdapter":
        """Register a mock LangChain tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for parameters
            handler: Function to call when tool is invoked

        Returns:
            Self for method chaining
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
            },
            "handler": handler or (lambda **kwargs: f"Mock result for {name}"),
        }
        logger.debug("Registered mock LangChain tool: %s", name)
        return self

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """List tools in LangChain/OpenAI format.

        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.tools.values()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a LangChain tool.

        Args:
            name: Tool name to execute
            arguments: Tool arguments
            **kwargs: Additional execution context

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool doesn't exist
        """
        if name not in self.tools:
            raise ValueError(f"Unknown LangChain tool: {name}")

        # Track call
        if name not in self.calls:
            self.calls[name] = []
        self.calls[name].append({"arguments": arguments, "kwargs": kwargs})

        # Execute handler
        handler = self.tools[name]["handler"]
        logger.debug("Calling mock LangChain tool '%s' with args: %s", name, arguments)

        if callable(handler):
            import inspect

            if inspect.iscoroutinefunction(handler):
                return await handler(**arguments)
            return handler(**arguments)
        return handler

    def was_called(self, name: str) -> bool:
        """Check if a tool was called."""
        return name in self.calls and len(self.calls[name]) > 0

    def call_count(self, name: str) -> int:
        """Get number of times a tool was called."""
        return len(self.calls.get(name, []))

    def get_calls(self, name: str) -> list[dict[str, Any]]:
        """Get all calls made to a tool."""
        return self.calls.get(name, [])

    def reset(self) -> None:
        """Clear call history."""
        self.calls.clear()

    def clear(self) -> None:
        """Clear tools and call history."""
        self.tools.clear()
        self.calls.clear()

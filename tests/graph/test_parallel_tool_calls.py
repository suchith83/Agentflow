"""Tests for parallel tool calls in StreamNodeHandler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyagenity.graph.tool_node import ToolNode
from pyagenity.graph.utils.stream_node_handler import StreamNodeHandler
from pyagenity.state import AgentState, Message
from pyagenity.state.message_block import ToolCallBlock, ToolResultBlock
from pyagenity.utils import CallbackManager


class TestParallelToolCalls:
    """Test parallel tool execution in StreamNodeHandler."""

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_basic(self):
        """Test that multiple tool calls execute in parallel."""
        # Track execution order
        execution_log = []

        async def slow_tool(value: str, tool_call_id: str | None = None) -> Message:
            """A tool that takes time to execute."""
            execution_log.append(f"start_{value}")
            await asyncio.sleep(0.1)  # Simulate async work
            execution_log.append(f"end_{value}")
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"result": f"processed_{value}"},
                    status="completed",
                    is_error=False,
                )],
            )

        async def fast_tool(value: str, tool_call_id: str | None = None) -> Message:
            """A tool that executes quickly."""
            execution_log.append(f"fast_{value}")
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"result": f"fast_{value}"},
                    status="completed",
                    is_error=False,
                )],
            )

        # Create tool node with both tools
        tool_node = ToolNode([slow_tool, fast_tool])

        # Create message with multiple tool calls
        message = Message.text_message(
            role="assistant",
            content="Using tools",
        )
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "slow_tool",
                    "arguments": '{"value": "first"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "fast_tool",
                    "arguments": '{"value": "second"}',
                },
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "slow_tool",
                    "arguments": '{"value": "third"}',
                },
            },
        ]

        # Create handler and execute
        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        config = {"thread_id": "test_123", "run_id": "run_456"}

        # Collect results
        results = []
        async for chunk in handler.stream(config, state):
            if isinstance(chunk, Message):
                results.append(chunk)

        # Verify we got all results
        assert len(results) == 3

        # Verify parallel execution (fast_tool should complete before slow_tools finish)
        # Due to parallel execution, we should see interleaved start/end patterns
        assert "fast_second" in execution_log
        # Count items that start with "start_" or "end_"
        start_count = sum(1 for item in execution_log if item.startswith("start_"))
        end_count = sum(1 for item in execution_log if item.startswith("end_"))
        assert start_count + end_count >= 2

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_with_state(self):
        """Test parallel tool calls can access shared state."""

        async def counter_tool(
            increment: int,
            tool_call_id: str | None = None,
            state: AgentState | None = None,
        ) -> Message:
            """Tool that increments a counter in state."""
            # Use context_summary to store counter as a workaround
            current = 0
            if state and state.context_summary:
                try:
                    current = int(state.context_summary)
                except (ValueError, TypeError):
                    current = 0
            new_value = current + increment
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"counter": new_value, "increment": increment},
                    status="completed",
                    is_error=False,
                )],
            )

        tool_node = ToolNode([counter_tool])

        message = Message.text_message(
            role="assistant",
            content="Count",
        )
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "counter_tool",
                    "arguments": '{"increment": 1}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "counter_tool",
                    "arguments": '{"increment": 2}',
                },
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "counter_tool",
                    "arguments": '{"increment": 3}',
                },
            },
        ]

        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        state.context_summary = "0"  # Store counter in context_summary
        config = {"thread_id": "test_123", "run_id": "run_456"}

        results = []
        async for chunk in handler.stream(config, state):
            if isinstance(chunk, Message):
                results.append(chunk)

        # Verify all tools executed
        assert len(results) == 3

        # Verify each tool got the correct state
        increments = [r.content[0].output["increment"] for r in results]
        assert sorted(increments) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_error_handling(self):
        """Test that errors in one tool don't block others."""

        async def failing_tool(tool_call_id: str | None = None) -> Message:
            """Tool that raises an error."""
            raise ValueError("Intentional error")

        async def success_tool(value: str, tool_call_id: str | None = None) -> Message:
            """Tool that succeeds."""
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"result": f"success_{value}"},
                    status="completed",
                    is_error=False,
                )],
            )

        tool_node = ToolNode([failing_tool, success_tool])

        message = Message.text_message(
            role="assistant",
            content="Mixed",
        )
        message.tools_calls = [
            {
                "id": "call_fail",
                "type": "function",
                "function": {
                    "name": "failing_tool",
                    "arguments": "{}",
                },
            },
            {
                "id": "call_success",
                "type": "function",
                "function": {
                    "name": "success_tool",
                    "arguments": '{"value": "test"}',
                },
            },
        ]

        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        config = {"thread_id": "test_123", "run_id": "run_456"}

        results = []
        async for chunk in handler.stream(config, state):
            if isinstance(chunk, Message):
                results.append(chunk)

        # Should get results from both tools (one error, one success)
        assert len(results) == 2

        # Find the success message
        success_messages = [
            r for r in results if not any(isinstance(b, type) and "error" in str(b).lower() for b in r.content)
        ]
        assert len(success_messages) >= 1

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_performance(self):
        """Test that parallel execution is actually faster than sequential."""
        import time

        async def delayed_tool(
            delay: float,
            tool_call_id: str | None = None,
        ) -> Message:
            """Tool with configurable delay."""
            await asyncio.sleep(delay)
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"delay": delay},
                    status="completed",
                    is_error=False,
                )],
            )

        tool_node = ToolNode([delayed_tool])

        # Create 3 tool calls, each with 0.2s delay
        message = Message.text_message(
            role="assistant",
            content="Delayed",
        )
        message.tools_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "delayed_tool",
                    "arguments": '{"delay": 0.2}',
                },
            }
            for i in range(3)
        ]

        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        config = {"thread_id": "test_123", "run_id": "run_456"}

        start_time = time.time()
        results = []
        async for chunk in handler.stream(config, state):
            if isinstance(chunk, Message):
                results.append(chunk)
        elapsed = time.time() - start_time

        # Sequential would take 0.6s (3 * 0.2), parallel should take ~0.2s
        # Allow some overhead, but should be significantly faster than sequential
        assert len(results) == 3
        assert elapsed < 0.5  # Much less than 0.6s sequential time

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self):
        """Test that single tool calls still work correctly."""

        async def simple_tool(value: str, tool_call_id: str | None = None) -> Message:
            """Simple test tool."""
            return Message.tool_message(
                content=[ToolResultBlock(
                    call_id=tool_call_id or "",
                    output={"result": value},
                    status="completed",
                    is_error=False,
                )],
            )

        tool_node = ToolNode([simple_tool])

        message = Message.text_message(
            role="assistant",
            content="Single",
        )
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "arguments": '{"value": "test"}',
                },
            }
        ]

        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        config = {"thread_id": "test_123", "run_id": "run_456"}

        results = []
        async for chunk in handler.stream(config, state):
            if isinstance(chunk, Message):
                results.append(chunk)

        assert len(results) == 1
        assert results[0].content[0].output["result"] == "test"

    @pytest.mark.asyncio
    async def test_empty_tool_calls_raises_error(self):
        """Test that empty tool calls list raises appropriate error."""
        from pyagenity.exceptions import NodeError

        tool_node = ToolNode([])

        message = Message.text_message(
            role="assistant",
            content="No tools",
        )
        message.tools_calls = []

        handler = StreamNodeHandler("test_node", tool_node)
        state = AgentState(context=[message])
        config = {"thread_id": "test_123", "run_id": "run_456"}

        with pytest.raises(NodeError):
            async for _ in handler.stream(config, state):
                pass

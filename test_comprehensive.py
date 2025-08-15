"""Comprehensive test for injectable parameters and thread safety."""

import asyncio
import time
from typing import Any

from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import (
    InjectCheckpointer,
    InjectConfig,
    InjectState,
    InjectStore,
    InjectToolCallID,
    Message,
)


# Test functions with injectable parameters
def sync_function(
    location: str,
    tool_call_id: InjectToolCallID = None,
    state: InjectState = None,
    config: InjectConfig = None,
) -> dict[str, Any]:
    """Sync function with injectable parameters."""
    return {
        "type": "sync",
        "location": location,
        "tool_call_id": tool_call_id,
        "state_has_context": hasattr(state, "context") if state else False,
        "config": config,
        "thread_id": id(time.thread_time()),
    }


async def async_function(
    location: str,
    tool_call_id: InjectToolCallID = None,
    checkpointer: InjectCheckpointer = None,
    store: InjectStore = None,
) -> dict[str, Any]:
    """Async function with injectable parameters."""
    await asyncio.sleep(0.1)  # Simulate async work
    return {
        "type": "async",
        "location": location,
        "tool_call_id": tool_call_id,
        "checkpointer": checkpointer,
        "store": store,
        "thread_id": id(time.thread_time()),
    }


def mixed_function(
    name: str,
    age: int = 25,
    tool_call_id: InjectToolCallID = None,
    regular_param: str = "default",
) -> dict[str, Any]:
    """Function with mixed regular and injectable parameters."""
    return {
        "name": name,
        "age": age,
        "tool_call_id": tool_call_id,
        "regular_param": regular_param,
    }


async def test_injectable_parameters():
    """Test injectable parameter functionality."""
    print("Testing Injectable Parameters...")

    # Create ToolNode with test functions
    node = ToolNode([sync_function, async_function, mixed_function])

    # Test 1: Check tool specs exclude injectable parameters
    tools = node.all_tools()
    print("\n1. Tool Specifications (should exclude injectable params):")

    for tool in tools:
        func_name = tool["function"]["name"]
        params = tool["function"]["parameters"]["properties"]
        print(f"  {func_name}: {list(params.keys())}")

    # Verify injectable parameters are excluded
    sync_params = tools[0]["function"]["parameters"]["properties"]
    assert "location" in sync_params  # Regular param should be included
    assert "tool_call_id" not in sync_params  # Injectable should be excluded
    assert "state" not in sync_params  # Injectable should be excluded
    assert "config" not in sync_params  # Injectable should be excluded

    async_params = tools[1]["function"]["parameters"]["properties"]
    assert "location" in async_params  # Regular param should be included
    assert "tool_call_id" not in async_params  # Injectable should be excluded
    assert "checkpointer" not in async_params  # Injectable should be excluded
    assert "store" not in async_params  # Injectable should be excluded

    mixed_params = tools[2]["function"]["parameters"]["properties"]
    assert "name" in mixed_params  # Regular param should be included
    assert "age" in mixed_params  # Regular param should be included
    assert "regular_param" in mixed_params  # Regular param should be included
    assert "tool_call_id" not in mixed_params  # Injectable should be excluded

    print("  ✓ Injectable parameters correctly excluded from tool specs")

    # Test 2: Execute functions with injectable parameters
    print("\n2. Function Execution with Injectable Parameters:")

    # Create test state
    state = AgentState()
    state.context = [Message.from_text("Test message")]

    # Test sync function
    sync_result = await node.execute(
        name="sync_function",
        args={"location": "Paris"},
        tool_call_id="test_call_001",
        config={"thread_id": "test_thread", "test": True},
        state=state,
    )
    print(f"  Sync result: {sync_result}")
    assert sync_result["location"] == "Paris"
    assert sync_result["tool_call_id"] == "test_call_001"
    assert sync_result["config"]["thread_id"] == "test_thread"

    # Test async function
    async_result = await node.execute(
        name="async_function",
        args={"location": "Tokyo"},
        tool_call_id="test_call_002",
        config={"thread_id": "test_thread"},
        state=state,
    )
    print(f"  Async result: {async_result}")
    assert async_result["location"] == "Tokyo"
    assert async_result["type"] == "async"

    # Test mixed function
    mixed_result = await node.execute(
        name="mixed_function",
        args={"name": "Alice", "regular_param": "custom"},
        tool_call_id="test_call_003",
        config={"thread_id": "test_thread"},
        state=state,
    )
    print(f"  Mixed result: {mixed_result}")
    assert mixed_result["name"] == "Alice"
    assert mixed_result["tool_call_id"] == "test_call_003"
    assert mixed_result["regular_param"] == "custom"

    print("  ✓ All function executions successful with injectable parameters")


async def test_thread_safety():
    """Test thread safety for concurrent execution."""
    print("\n3. Thread Safety Test:")

    node = ToolNode([sync_function, async_function])
    state = AgentState()
    state.context = [Message.from_text("Test message")]

    # Create multiple concurrent executions
    tasks = []

    # Add sync function tasks
    for i in range(5):
        task = node.execute(
            name="sync_function",
            args={"location": f"City_{i}"},
            tool_call_id=f"sync_call_{i}",
            config={"task_id": i},
            state=state,
        )
        tasks.append(task)

    # Add async function tasks
    for i in range(5):
        task = node.execute(
            name="async_function",
            args={"location": f"AsyncCity_{i}"},
            tool_call_id=f"async_call_{i}",
            config={"task_id": i + 5},
            state=state,
        )
        tasks.append(task)

    # Execute all tasks concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    print(f"  Executed {len(tasks)} tasks in {end_time - start_time:.2f} seconds")

    # Verify results
    sync_results = results[:5]
    async_results = results[5:]

    for i, result in enumerate(sync_results):
        assert result["location"] == f"City_{i}"
        assert result["tool_call_id"] == f"sync_call_{i}"
        assert result["type"] == "sync"

    for i, result in enumerate(async_results):
        assert result["location"] == f"AsyncCity_{i}"
        assert result["tool_call_id"] == f"async_call_{i}"
        assert result["type"] == "async"

    print("  ✓ All concurrent executions completed successfully")
    print("  ✓ Thread safety maintained for both sync and async functions")


async def main():
    """Run all tests."""
    print("PyAgenity Injectable Parameters & Thread Safety Test")
    print("=" * 60)

    try:
        await test_injectable_parameters()
        await test_thread_safety()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✓ Injectable parameters working correctly")
        print("✓ Tool specifications exclude injectable params")
        print("✓ Function execution with injection successful")
        print("✓ Thread safety maintained for concurrent execution")
        print("✓ Both sync and async functions supported")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

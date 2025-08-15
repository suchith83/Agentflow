"""Example demonstrating injectable parameter types."""

import json
from typing import Any

from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.utils import (
    InjectCheckpointer,
    InjectConfig,
    InjectState,
    InjectStore,
    InjectToolCallID,
)


def get_current_weather(
    location: str,
    unit: str = "celsius",
    times: list[str] | None = None,
    tool_call_id: InjectToolCallID = None,  # This will be injected automatically
    state: InjectState = None,  # This will be injected automatically
) -> dict[str, Any]:
    """Get the current weather in a given location.

    The tool_call_id and state parameters will be automatically injected
    and won't appear in the LLM tool specification.
    """
    # Access injected parameters
    print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")

    # Dummy implementation for demo
    temp = 20 if unit == "celsius" else 68
    return {"location": location, "temperature": temp, "unit": unit}


def advanced_weather_function(
    location: str,
    checkpointer: InjectCheckpointer = None,  # Auto-injected
    store: InjectStore = None,  # Auto-injected
    config: InjectConfig = None,  # Auto-injected
) -> dict[str, Any]:
    """More advanced weather function showing all injectable types."""
    print(f"Checkpointer available: {checkpointer is not None}")
    print(f"Store available: {store is not None}")
    print(f"Config: {config}")

    return {"location": location, "advanced": True}


async def async_weather_function(
    location: str,
    forecast_days: int = 1,
    tool_call_id: InjectToolCallID = None,
) -> dict[str, Any]:
    """Async weather function to test async/sync compatibility."""
    print(f"Async weather function called with tool_call_id: {tool_call_id}")
    return {"location": location, "forecast_days": forecast_days, "async": True}


if __name__ == "__main__":
    # Create ToolNode with functions that have injectable parameters
    node = ToolNode([get_current_weather, advanced_weather_function, async_weather_function])

    # Get tool specifications - injectable parameters should be excluded
    tools = node.all_tools()
    print("TOOLS SPEC (Injectable parameters should be excluded):")
    print(json.dumps(tools, indent=2))

    print("\n" + "=" * 50)
    print("Notice that injectable parameters like 'tool_call_id', 'state',")
    print("'checkpointer', 'store', and 'config' are NOT in the tool spec!")
    print("=" * 50)

    # Test execution (would normally be called by the framework)
    print("\nTEST EXECUTION:")

    # Note: In a real scenario, these would be called by the framework during execution
    # Here we're just demonstrating the interface

    # Simulate execution parameters
    tool_args = {"location": "New York", "unit": "fahrenheit"}
    tool_call_id = "test_call_123"
    config = {"thread_id": "test_thread"}

    print(f"Would execute get_current_weather with args: {tool_args}")
    print(f"Injectable parameters would be: tool_call_id={tool_call_id}, config={config}")

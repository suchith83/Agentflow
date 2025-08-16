"""Demo showing enhanced injectable types with proper IDE type hints."""

from typing import Any, TypedDict

from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import (
    InjectCheckpointer,
    InjectConfig,
    InjectState,
    InjectStore,
    InjectToolCallID,
)


# Example of custom state class that users might define
class MyCustomState:
    """Custom state class with specific attributes."""

    def __init__(self):
        self.context: list[str] = []
        self.user_preferences: dict[str, Any] = {}
        self.session_id: str = ""


# Example of typed configuration
class MyConfigDict(TypedDict):
    """Typed configuration dictionary."""

    thread_id: str
    max_retries: int
    debug_mode: bool


def basic_weather_function(
    location: str,
    tool_call_id: InjectToolCallID = None,  # IDE knows this is str
    state: InjectState = None,  # IDE knows this is AgentState
) -> dict[str, Any]:
    """Basic weather function with default injectable types."""
    return {
        "location": location,
        "tool_call_id": tool_call_id,  # IDE: str
        "state_type": type(state).__name__,  # IDE: AgentState
    }


def enhanced_weather_function(
    location: str,
    tool_call_id: InjectToolCallID[str] = None,  # Explicit: IDE knows this is str
    state: InjectState[MyCustomState] = None,  # IDE knows this is MyCustomState
    config: InjectConfig[MyConfigDict] = None,  # IDE knows this is MyConfigDict
) -> dict[str, Any]:
    """Enhanced weather function with explicit generic types."""
    # IDE will now provide autocomplete for these!
    if state:
        # IDE knows: state.context is list[str]
        # IDE knows: state.user_preferences is dict[str, Any]
        # IDE knows: state.session_id is str
        context_length = len(state.context)
        session = state.session_id
    else:
        context_length = 0
        session = "unknown"

    if config:
        # IDE knows: config["thread_id"] is str
        # IDE knows: config["max_retries"] is int
        # IDE knows: config["debug_mode"] is bool
        thread_id = config["thread_id"]
        debug = config["debug_mode"]
    else:
        thread_id = "unknown"
        debug = False

    return {
        "location": location,
        "tool_call_id": tool_call_id,
        "context_length": context_length,
        "session_id": session,
        "thread_id": thread_id,
        "debug_mode": debug,
    }


async def async_weather_with_types(
    location: str,
    forecast_days: int = 7,
    state: InjectState[AgentState] = None,  # Explicit AgentState type
    checkpointer: InjectCheckpointer = None,  # Default checkpointer type
    store: InjectStore = None,  # Default store type
) -> dict[str, Any]:
    """Async function showing mixed explicit and default types."""
    # IDE knows state is AgentState and can provide autocomplete
    if state and hasattr(state, "context"):
        message_count = len(state.context)
    else:
        message_count = 0

    return {
        "location": location,
        "forecast_days": forecast_days,
        "message_count": message_count,
        "has_checkpointer": checkpointer is not None,
        "has_store": store is not None,
    }


if __name__ == "__main__":
    # Create ToolNode with the enhanced functions
    node = ToolNode([basic_weather_function, enhanced_weather_function, async_weather_with_types])

    # Check tool specifications - injectable params should still be excluded
    tools = node.all_tools()
    print("Enhanced Injectable Types Demo")
    print("=" * 50)
    print("\nTool Specifications (injectable params excluded):")
    for tool in tools:
        func_name = tool["function"]["name"]
        params = list(tool["function"]["parameters"]["properties"].keys())
        print(f"  {func_name}: {params}")

    print("\n" + "=" * 50)
    print("Benefits of Enhanced Injectable Types:")
    print("✓ IDE provides proper type hints and autocomplete")
    print("✓ Type safety for injected parameters")
    print("✓ Clear documentation of expected types")
    print("✓ Backward compatibility maintained")
    print("✓ Injectable parameters still excluded from LLM specs")

    print("\nExample Usage in IDE:")
    print("  state: InjectState[MyCustomState] = None")
    print("  # IDE now knows:")
    print("  #   state.context -> list[str]")
    print("  #   state.user_preferences -> dict[str, Any]")
    print("  #   state.session_id -> str")

    print("\n  config: InjectConfig[MyConfigDict] = None")
    print("  # IDE now knows:")
    print("  #   config['thread_id'] -> str")
    print("  #   config['max_retries'] -> int")
    print("  #   config['debug_mode'] -> bool")

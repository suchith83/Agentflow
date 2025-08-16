from pyagenity.graph.graph.tool_node import ToolNode


def current_weather(location: str) -> dict:
    """Fetch the current weather for a specific location."""
    # Simulated weather data
    return {
        "location": location,
        "temperature": 72,
        "condition": "Sunny",
    }


tool_node = ToolNode([current_weather])

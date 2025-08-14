from typing import List, Optional
from pyagenity.graph.graph.tool_node import ToolNode
import json


def get_current_weather(
    location: str,
    unit: str = "celsius",
    times: Optional[List[str]] = None,
):
    """Get the current weather in a given location.

    Example description used by LLMs.
    """
    # Dummy implementation for demo
    temp = 20 if unit == "celsius" else 68
    return {"location": location, "temperature": temp, "unit": unit}


if __name__ == "__main__":
    node = ToolNode([get_current_weather])
    tools = node.all_tools()
    print("TOOLS SPEC:")
    print(json.dumps(tools, indent=2))

    print("\nEXECUTE DEMO:")
    result = node.execute("get_current_weather", {"location": 2})
    print(json.dumps(result, indent=2))

"""Example: Using Agent with Google Gemini in a StateGraph.

This example demonstrates a complete workflow using Google's Gemini model
with the Agent class, including tools and conditional routing.

Requirements:
    pip install 10xscale-agentflow[litellm]
    export GEMINI_API_KEY=your_api_key_here
"""

import asyncio
import os

from agentflow.graph import Agent, StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils import END


# Define some tools for the agent
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. "San Francisco, CA"
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information as a string
    """
    # Simulated weather data
    weather_data = {
        "San Francisco, CA": {"celsius": 18, "fahrenheit": 64, "condition": "Foggy"},
        "New York, NY": {"celsius": 22, "fahrenheit": 72, "condition": "Sunny"},
        "London, UK": {"celsius": 15, "fahrenheit": 59, "condition": "Rainy"},
        "Tokyo, Japan": {"celsius": 25, "fahrenheit": 77, "condition": "Clear"},
    }

    data = weather_data.get(location, {"celsius": 20, "fahrenheit": 68, "condition": "Unknown"})
    temp = data[unit]
    condition = data["condition"]

    return (
        f"The weather in {location} is {condition} with a temperature of {temp}°{unit[0].upper()}"
    )


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of the calculation
    """
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def search_wiki(query: str) -> str:
    """Search for information (simulated).

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulated wiki data
    wiki_data = {
        "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "agent": "An agent is an autonomous entity that observes and acts upon an environment.",
    }

    for key, value in wiki_data.items():
        if key.lower() in query.lower():
            return value

    return f"No information found for '{query}'"


async def main():
    """Run the Google Gemini Agent example."""
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("   Get your key from: https://makersuite.google.com/app/apikey")
        return

    print("🚀 Google Gemini Agent Example")
    print("=" * 70)

    # Create tools
    tools = [get_current_weather, calculate, search_wiki]
    tool_node = ToolNode(tools)

    # Create agent with Gemini
    agent = Agent(
        model="gemini/gemini-2.0-flash-exp",  # LiteLLM format for Gemini
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to tools. "
                    "Use the available tools to answer questions accurately. "
                    "Always be concise and friendly."
                ),
            }
        ],
        tools=tools,
    )

    # Define routing logic
    def should_continue(state: AgentState) -> str:
        """Determine next step based on last message."""
        if not state.context:
            return END

        last_message = state.context[-1]

        # If the last message has tool calls, execute them
        if hasattr(last_message, "tools_calls") and last_message.tools_calls:
            return "tools"

        # Otherwise, we're done
        return END

    # Build the graph
    graph = StateGraph()
    graph.add_node("agent", agent)  # Agent as a node!
    graph.add_node("tools", tool_node)

    # Setup flow
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph.add_edge("tools", "agent")  # After tools, go back to agent

    # Compile the graph
    compiled = graph.compile()

    # Test queries
    test_queries = [
        "What's the weather in Tokyo, Japan?",
        "Calculate 15 * 23 + 47",
        "Tell me about Python programming language",
        "What's 100 divided by 4, and then what's the weather in London?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 70)

        # Create initial state
        state = AgentState()
        state.context = [
            Message(
                id=str(i),
                role="user",
                content=query,
            )
        ]

        # Execute the graph
        result = await compiled.ainvoke({"state": state})

        # Display the conversation
        final_state = result["state"]

        # Show tool calls if any
        tool_calls_found = False
        for msg in final_state.context:
            if hasattr(msg, "tools_calls") and msg.tools_calls:
                tool_calls_found = True
                print(f"🔧 Tools used: {len(msg.tools_calls)}")
                for tool_call in msg.tools_calls:
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    print(f"   - {tool_name}")

        # Show final response
        last_message = final_state.context[-1]
        if hasattr(last_message, "text"):
            response = last_message.text()
        else:
            response = str(last_message.content)

        print(f"\n💬 Response: {response}")

    # Test streaming
    print("\n" + "=" * 70)
    print("🔄 Testing Streaming Mode")
    print("=" * 70)

    query = "What's the weather in New York and calculate 50 * 2?"
    print(f"\n📝 Query: {query}")
    print("-" * 70)

    state = AgentState()
    state.context = [
        Message(
            id="stream_test",
            role="user",
            content=query,
        )
    ]

    print("Streaming response:")
    chunk_count = 0
    async for chunk in compiled.astream({"state": state}):
        chunk_count += 1
        # Show chunk types
        if hasattr(chunk, "event"):
            print(f"  [{chunk_count}] {chunk.event}")

    print(f"\n✅ Streamed {chunk_count} chunks")

    # Cleanup
    await compiled.aclose()

    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

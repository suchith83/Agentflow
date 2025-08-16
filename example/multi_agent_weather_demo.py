"""
Multi-agent weather demo with play/pause and checkpointer.
- Main agent: handles general conversation, delegates weather queries
- Weather agent: handles weather queries, uses weather tool
- Weather tool: returns fake weather data
"""

import asyncio

from pyagenity.graph.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message


# --- Weather Tool ---
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Fake weather tool: returns dummy weather data."""
    temp = 25 if unit == "celsius" else 77
    return {"location": location, "temperature": temp, "unit": unit}


weather_tool_node = ToolNode([get_weather])


# --- Weather Agent Node ---
async def weather_agent_node(state: AgentState, config: dict, checkpointer=None, store=None):
    last_msg = state.context[-1].content if state.context else ""
    # Parse location from message (simple demo logic)
    if "weather" in last_msg.lower():
        location = "Dhaka" if "dhaka" in last_msg.lower() else "Unknown"
        weather = await weather_tool_node.execute(
            "get_weather",
            {"location": location},
            tool_call_id="1",
            config=config,
            state=state,
            checkpointer=checkpointer,
            store=store,
        )
        reply = f"Weather in {weather['location']}: {weather['temperature']}°{weather['unit']}"
    else:
        reply = "I can only answer weather questions."
    return Message.from_text(reply)


# --- Main Agent Node ---
async def main_agent_node(state: AgentState, config: dict, checkpointer=None, store=None):
    last_msg = state.context[-1].content if state.context else ""
    if "weather" in last_msg.lower():
        # Route to weather agent
        return Message.from_text("Routing to weather agent...")
    else:
        return Message.from_text(f"General reply: {last_msg}")


# --- Router Node ---
def router_node(state: AgentState, config: dict, checkpointer=None, store=None):
    last_msg = state.context[-1].content if state.context else ""
    if "weather" in last_msg.lower():
        return "weather_agent"
    else:
        return "main_agent"


async def main():
    print("=== Multi-Agent Weather Demo (with play/pause) ===\n")
    graph = StateGraph()
    graph.add_node("main_agent", main_agent_node)
    graph.add_node("weather_agent", weather_agent_node)
    graph.add_node("router", router_node)
    graph.add_edge("router", "main_agent")
    graph.add_edge("router", "weather_agent")
    graph.add_edge("main_agent", "router")
    graph.add_edge("weather_agent", "router")
    graph.set_entry_point("router")

    checkpointer = InMemoryCheckpointer()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["main_agent", "weather_agent"],
    )

    # Simulate 5 rounds
    user_inputs = [
        "Hello!",
        "Can you tell me the weather?",
        "What's the weather in Dhaka?",
        "Thanks!",
        "Bye.",
    ]
    config = {"thread_id": "demo_user"}
    for i, user_input in enumerate(user_inputs):
        print(f"\n--- Round {i + 1} ---")
        input_data = {"messages": [Message.from_text(user_input)]}
        result = await compiled.ainvoke(input_data=input_data, config=config)
        messages = result.get("messages", [])
        for msg in messages:
            print(f"Agent: {msg.content}")
        # Pause/resume: state is persisted in checkpointer
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

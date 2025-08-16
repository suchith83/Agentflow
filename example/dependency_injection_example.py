"""
Comprehensive example demonstrating PyAgenity's dependency injection
and generic state management features.

This example shows:
1. Custom AgentState subclass with additional fields
2. Dependency injection for reusable components
3. Generic type support throughout the system
4. Custom tools with dependency injection
"""

from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from litellm import completion

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import (
    END,
    DependencyContainer,
    InjectDep,
    InjectState,
    InjectToolCallID,
    Message,
    convert_messages,
)


load_dotenv()


# === 1. Custom State Class ===
@dataclass
class CustomAgentState(AgentState):
    """Custom state that extends AgentState with application-specific fields."""

    user_preferences: dict[str, Any] = field(default_factory=dict)
    session_data: dict[str, str] = field(default_factory=dict)
    analytics: dict[str, int] = field(default_factory=lambda: {"api_calls": 0, "tool_uses": 0})


# === 2. Injectable Dependencies ===
class DatabaseService:
    """Mock database service for demonstration."""

    def __init__(self):
        self.data = {
            "users": {"john": {"preferences": {"theme": "dark", "lang": "en"}}},
            "products": {"laptop": {"price": 999, "stock": 5}},
        }

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        return self.data.get("users", {}).get(user_id, {}).get("preferences", {})

    def get_product_info(self, product_id: str) -> dict[str, Any]:
        return self.data.get("products", {}).get(product_id, {})


class LoggingService:
    """Mock logging service for demonstration."""

    def __init__(self):
        self.logs = []

    def log(self, level: str, message: str, **kwargs):
        entry = {
            "level": level,
            "message": message,
            "timestamp": "2024-01-01T00:00:00Z",  # Mock timestamp
            **kwargs,
        }
        self.logs.append(entry)
        print(f"[{level.upper()}] {message}")

    def info(self, message: str, **kwargs):
        self.log("info", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("error", message, **kwargs)


class CacheService:
    """Mock cache service for demonstration."""

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 300):
        self.cache[key] = value
        return True


# === 3. Tools with Dependency Injection ===
def get_user_preferences(
    user_id: str,
    tool_call_id: InjectToolCallID[str] = None,
    state: InjectState[CustomAgentState] = None,
    database: InjectDep[DatabaseService] = None,
    logger: InjectDep[LoggingService] = None,
) -> str:
    """Get user preferences from database with dependency injection."""

    if logger:
        logger.info(f"Fetching preferences for user: {user_id}")

    if database:
        preferences = database.get_user_preferences(user_id)

        # Update state with user preferences
        if state:
            state.user_preferences.update(preferences)
            state.analytics["api_calls"] += 1

        if logger:
            logger.info(f"Retrieved preferences: {preferences}")

        return f"User {user_id} preferences: {preferences}"

    return "Database service not available"


def get_product_info(
    product_id: str,
    tool_call_id: InjectToolCallID[str] = None,
    state: InjectState[CustomAgentState] = None,
    database: InjectDep[DatabaseService] = None,
    cache: InjectDep[CacheService] = None,
    logger: InjectDep[LoggingService] = None,
) -> str:
    """Get product information with caching and dependency injection."""

    if logger:
        logger.info(f"Fetching product info for: {product_id}")

    # Try cache first
    cache_key = f"product:{product_id}"
    if cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            if logger:
                logger.info(f"Cache hit for {product_id}")
            return cached_result

    # Fetch from database
    if database:
        product_info = database.get_product_info(product_id)

        # Update analytics
        if state:
            state.analytics["api_calls"] += 1
            state.analytics["tool_uses"] += 1

        result = f"Product {product_id}: {product_info}"

        # Cache the result
        if cache:
            cache.set(cache_key, result)

        if logger:
            logger.info(f"Retrieved and cached product info: {product_info}")

        return result

    return "Database service not available"


# === 4. Main Agent Function ===
def main_agent(
    state: CustomAgentState,
    config: dict[str, Any],
    checkpointer=None,
    store=None,
    logger: InjectDep[LoggingService] = None,
) -> Any:
    """Main agent with dependency injection."""

    if logger:
        logger.info("Main agent starting", step=state.execution_meta.step)

    # Update session data
    state.session_data["last_agent_call"] = "main_agent"

    prompts = """
    You are a helpful assistant with access to user preferences and product information.
    You can help users with their preferences and provide product details.

    Available tools:
    - get_user_preferences: Get user's saved preferences
    - get_product_info: Get detailed product information
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    # Check if the last message is a tool result
    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        # Make final response without tools since we just got tool results
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        # Regular response with tools available
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tool_node.all_tools(),
        )

    # Update analytics
    state.analytics["api_calls"] += 1

    if logger:
        logger.info("Main agent completed", analytics=state.analytics)

    return response


def should_use_tools(state: CustomAgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done
    if last_message.role == "tool" and last_message.tool_call_id is not None:
        return END

    return END


# === 5. Setup and Execution ===
def create_dependency_injection_example():
    """Create and configure the dependency injection example."""

    # Create dependency container
    container = DependencyContainer()

    # Register dependencies
    container.register("database", DatabaseService())
    container.register("logger", LoggingService())
    container.register("cache", CacheService())

    # Create custom state instance
    custom_state = CustomAgentState(
        user_preferences={"user_id": "john"},
        session_data={"session_id": "sess_123"},
    )

    # Create tool node with dependency injection support
    global tool_node
    tool_node = ToolNode([get_user_preferences, get_product_info])

    # Build the graph with dependency injection
    graph = StateGraph[CustomAgentState](
        state=custom_state,
        dependency_container=container,
    )

    graph.add_node("MAIN", main_agent)
    graph.add_node("TOOL", tool_node)

    # Add conditional edges from MAIN
    graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})

    # Always go back to MAIN after TOOL execution
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph


if __name__ == "__main__":
    print("PyAgenity Dependency Injection & Generic State Example")
    print("=" * 60)

    # Create the graph
    graph = create_dependency_injection_example()
    app = graph.compile()

    # Show dependency information
    print("\\nRegistered Dependencies:")
    deps = graph.dependency_container.list_dependencies()
    for dep in deps:
        print(f"  - {dep}: {type(graph.dependency_container.get(dep)).__name__}")

    # Show tool specifications (dependencies excluded from LLM tools)
    tools = tool_node.all_tools()
    print("\\nTool Specifications for LLM:")
    for tool in tools:
        func_name = tool["function"]["name"]
        params = list(tool["function"]["parameters"]["properties"].keys())
        print(f"  - {func_name}: {params}")

    # Run the example
    print("\\nRunning example...")

    config = {"thread_id": "demo_123", "recursion_limit": 10}

    # Test 1: Get user preferences
    inp1 = {"messages": [Message.from_text("Hi! Can you get my preferences? My user ID is john.")]}

    result1 = app.invoke(inp1, config=config)
    print(f"\\nResult 1: {len(result1['messages'])} messages")

    # Test 2: Get product information
    inp2 = {"messages": [Message.from_text("Can you tell me about the laptop product?")]}

    result2 = app.invoke(inp2, config=config)
    print(f"Result 2: {len(result2['messages'])} messages")

    # Show final state
    final_state = result2.get("state")
    if isinstance(final_state, CustomAgentState):
        print("\\nFinal State Analytics:")
        print(f"  API calls: {final_state.analytics['api_calls']}")
        print(f"  Tool uses: {final_state.analytics['tool_uses']}")
        print(f"  User preferences: {final_state.user_preferences}")
        print(f"  Session data: {final_state.session_data}")

    # Show logs
    logger = graph.dependency_container.get("logger")
    print(f"\\nTotal log entries: {len(logger.logs)}")

    print("\\nExample completed successfully!")

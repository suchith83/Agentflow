"""
Example: Graceful Shutdown with Signal Handling

This example demonstrates how to implement graceful shutdown in a long-running
Agentflow application with proper signal handling for SIGTERM and SIGINT.

This example uses a realistic React agent with tool calling to demonstrate
graceful shutdown in a production-like scenario.
"""

import asyncio
import datetime
import logging
import sys

from litellm import acompletion

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils import END
from agentflow.utils.converter import convert_messages
from agentflow.utils.shutdown import GracefulShutdownManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define tools for the agent
def get_current_time(tool_call_id: str | None = None) -> str:
    """Get the current time."""
    return f"Current time is {datetime.datetime.now().strftime('%H:%M:%S')}"


def get_system_status(tool_call_id: str | None = None) -> str:
    """Get the system status."""
    return "System status: All services operational"


def calculate(expression: str, tool_call_id: str | None = None) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow basic math operations for safety
        # Note: In production, use a proper math parser instead of eval
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


# Create tool node with available tools
tool_node = ToolNode([get_current_time, get_system_status, calculate])


# Define the main agent node
async def main_agent(state: AgentState) -> ModelResponseConverter:
    """Main agent that reasons and decides when to use tools."""
    system_prompt = """You are a helpful assistant with access to tools.
You can check the time, system status, and perform calculations.
Use tools when appropriate to help answer user questions."""

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": system_prompt}],
        state=state,
    )

    # Check if last message is a tool result
    needs_tools = not (state.context and state.context[-1].role == "tool")

    if needs_tools:
        # Include tools in the request
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.0-flash-exp",
            messages=messages,
            tools=tools,
        )
    else:
        # Final response without tools
        response = await acompletion(
            model="gemini/gemini-2.0-flash-exp",
            messages=messages,
        )

    return ModelResponseConverter(response, converter="litellm")


def route_decision(state: AgentState) -> str:
    """Route to tool execution or end based on agent output."""
    if not state.context:
        return "TOOL"

    last_message = state.context[-1]

    # Check if assistant made tool calls
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is tool result, go back to agent
    if last_message.role == "tool":
        return "MAIN"

    # Otherwise, we're done
    return END


def build_graph() -> StateGraph:
    """Build a React agent graph with tool calling."""
    graph = StateGraph()
    graph.add_node("MAIN", main_agent)
    graph.add_node("TOOL", tool_node)

    # Conditional routing from main agent
    graph.add_conditional_edges("MAIN", route_decision, {"TOOL": "TOOL", END: END, "MAIN": "MAIN"})

    # Always return to main after tool execution
    graph.add_edge("TOOL", "MAIN")

    graph.set_entry_point("MAIN")
    return graph


async def long_running_service():
    """
    Long-running service that processes tasks until shutdown signal received.

    This demonstrates:
    - Graceful shutdown with signal handling
    - Protected initialization and cleanup
    - Proper resource management
    - Shutdown statistics logging
    - React agent with real tool calling
    """
    # Configuration
    SHUTDOWN_TIMEOUT = 30.0

    # Create shutdown manager
    shutdown_manager = GracefulShutdownManager(shutdown_timeout=SHUTDOWN_TIMEOUT)

    logger.info("Building and compiling graph...")
    graph = build_graph().compile(shutdown_timeout=SHUTDOWN_TIMEOUT)

    # Register signal handlers for SIGTERM and SIGINT
    shutdown_manager.register_signal_handlers()
    logger.info("Signal handlers registered (Ctrl+C to stop)")

    task_count = 0  # Initialize task counter before try block
    try:
        # Protected initialization
        logger.info("Starting initialization (protected from interruption)...")
        with shutdown_manager.protect_section():
            await asyncio.sleep(2)  # Simulate initialization
            logger.info("Initialization complete")

        # Main processing loop
        logger.info("Entering main loop. Press Ctrl+C to shutdown gracefully...")

        # Define sample queries for the agent
        sample_queries = [
            "What time is it?",
            "Can you calculate 15 + 27?",
            "What's the system status?",
            "Calculate 100 * 5 and tell me the time",
        ]

        while not shutdown_manager.shutdown_requested:
            try:
                # Check for shutdown every 1 second
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=1.0)

                # Process a task
                task_count += 1
                query = sample_queries[(task_count - 1) % len(sample_queries)]
                logger.info(f"Processing task #{task_count}: {query}")

                result = await graph.ainvoke(
                    {"messages": [Message.text_message(query, role="user")]},
                    config={"thread_id": f"thread_{task_count}"},
                )

                # Log the final response
                if result.get("messages"):
                    last_msg = result["messages"][-1]
                    if last_msg.role == "assistant":
                        logger.info(f"Agent response: {last_msg.content[:100]}...")

                logger.info(f"Task #{task_count} completed")

                # Simulate some delay between tasks
                await asyncio.sleep(2)

            except TimeoutError:
                # No task available, continue to check shutdown flag
                continue
            except Exception as e:
                logger.exception("Error processing task: %s", e)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt (Ctrl+C)")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
    finally:
        # Protected cleanup
        logger.info("Starting cleanup (protected from interruption)...")
        with shutdown_manager.protect_section():
            # Close graph with detailed statistics
            stats = await graph.aclose()

            # Log shutdown statistics
            logger.info("=== Shutdown Statistics ===")
            logger.info(f"Total duration: {stats.get('total_duration', 0):.2f}s")
            logger.info(f"Background tasks: {stats.get('background_tasks', {})}")
            logger.info(f"Checkpointer: {stats.get('checkpointer', {})}")
            logger.info(f"Publisher: {stats.get('publisher', {})}")
            logger.info(f"Store: {stats.get('store', {})}")

            # Unregister signal handlers
            shutdown_manager.unregister_signal_handlers()
            logger.info("Cleanup complete")

        logger.info(f"Processed {task_count} tasks total")
        logger.info("Application shutdown complete")


async def main():
    """Main entry point."""
    logger.info("=== Graceful Shutdown Example ===")
    logger.info("This example demonstrates graceful shutdown with signal handling.")
    logger.info("Press Ctrl+C at any time to trigger graceful shutdown.")
    logger.info("")

    try:
        await long_running_service()
    except KeyboardInterrupt:
        logger.info("Application terminated")
    finally:
        logger.info("Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)

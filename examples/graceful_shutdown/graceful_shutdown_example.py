"""
Example: Graceful Shutdown with Signal Handling

This example demonstrates how to implement graceful shutdown in a long-running
Agentflow application with proper signal handling for SIGTERM and SIGINT.
"""

import asyncio
import logging
import sys
from typing import Any

from agentflow.graph import StateGraph
from agentflow.state import AgentState, Message
from agentflow.utils import END
from agentflow.utils import GracefulShutdownManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define a simple agent node
async def process_node(state: AgentState, config: dict[str, Any]) -> AgentState:
    """Process messages - simulates some work."""
    logger.info("Processing node executing...")
    await asyncio.sleep(1)  # Simulate processing time

    messages = state.context
    last_message = messages[-1] if messages else None

    if last_message and last_message.role == "user":
        response = Message.text_message(f"Processed: {last_message.content}", role="assistant")
        state.context.append(response)

    return state


def build_graph() -> StateGraph:
    """Build a simple graph for demonstration."""
    graph = StateGraph()
    graph.add_node("process", process_node)
    graph.set_entry_point("process")
    graph.add_edge("process", END)
    return graph


async def long_running_service():
    """
    Long-running service that processes tasks until shutdown signal received.

    This demonstrates:
    - Graceful shutdown with signal handling
    - Protected initialization and cleanup
    - Proper resource management
    - Shutdown statistics logging
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

    try:
        # Protected initialization
        logger.info("Starting initialization (protected from interruption)...")
        with shutdown_manager.protect_section():
            await asyncio.sleep(2)  # Simulate initialization
            logger.info("Initialization complete")

        # Main processing loop
        logger.info("Entering main loop. Press Ctrl+C to shutdown gracefully...")
        task_count = 0

        while not shutdown_manager.shutdown_requested:
            try:
                # Check for shutdown every 1 second
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=1.0)

                # Process a task
                task_count += 1
                logger.info(f"Processing task #{task_count}")

                result = await graph.ainvoke(
                    {"messages": [Message.text_message(f"Task #{task_count}", role="user")]}
                )

                logger.info(f"Task #{task_count} completed: {result}")

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
        print("\nShutdown complete")
        sys.exit(0)

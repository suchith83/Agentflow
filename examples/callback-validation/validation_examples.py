"""
Comprehensive examples showing how to use the PyAgenity callback system
for custom validation logic across AI, TOOL, and MCP invocations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from pyagenity.graph import StateGraph
from pyagenity.state import AgentState, Message
from pyagenity.utils import (
    AfterInvokeCallback,
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
    OnErrorCallback,
)


# Configure logging for examples
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_ERROR_COUNT = 10


class SecurityValidationCallback(BeforeInvokeCallback[CallbackContext, bool]):
    """Example security validation for AI and tool calls."""

    async def execute(self, context: CallbackContext) -> bool:
        """
        Validate calls based on security policies.
        Returns True to allow, False to block.
        """
        logger.info(
            f"🛡️  Security check for {context.invocation_type.value} call: {context.function_name}"
        )

        # Block dangerous function names
        dangerous_functions = ["delete_all", "format_disk", "rm_rf"]
        if context.function_name in dangerous_functions:
            logger.warning(f"❌ Blocked dangerous function: {context.function_name}")
            return False

        # Block AI calls with sensitive keywords
        if (
            context.invocation_type == InvocationType.AI
            and context.metadata
            and "input" in context.metadata
        ):
            sensitive_keywords = ["password", "secret", "api_key"]
            input_text = str(context.metadata["input"]).lower()
            if any(keyword in input_text for keyword in sensitive_keywords):
                logger.warning("❌ Blocked AI call with sensitive content")
                return False

        # Rate limiting for MCP calls
        if context.invocation_type == InvocationType.MCP:
            # In real implementation, you'd check a rate limiter
            logger.info(f"✅ MCP call allowed: {context.function_name}")

        logger.info(f"✅ Security check passed for {context.function_name}")
        return True


class AuditLoggingCallback(AfterInvokeCallback[CallbackContext, Any, None]):
    """Example audit logging for all successful invocations."""

    def __init__(self):
        self.audit_log: list[dict[str, Any]] = []

    async def execute(self, context: CallbackContext, result: Any) -> None:
        """Log successful invocations for audit purposes."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "invocation_type": context.invocation_type.value,
            "node_name": context.node_name,
            "function_name": context.function_name,
            "success": True,
            "result_type": type(result).__name__,
            "metadata": context.metadata,
        }

        self.audit_log.append(audit_entry)
        logger.info(
            f"📝 Audit logged: {context.invocation_type.value} call to {context.function_name}"
        )

        # In real implementation, you might save to database or file

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log entries."""
        return self.audit_log.copy()


class ErrorHandlingCallback(OnErrorCallback[CallbackContext, Exception, None]):
    """Example error handling and recovery logic."""

    def __init__(self):
        self.error_count = 0
        self.max_retries = 3

    async def execute(self, context: CallbackContext, error: Exception) -> None:
        """Handle errors with custom logic."""
        self.error_count += 1

        logger.error(f"❌ Error in {context.invocation_type.value} call: {error}")

        # Different error handling strategies
        if context.invocation_type == InvocationType.AI:
            logger.info("🔄 AI call failed - could implement retry with backoff")

        elif context.invocation_type == InvocationType.TOOL:
            logger.info("🛠️  Tool call failed - could try alternative tool")

        elif context.invocation_type == InvocationType.MCP:
            logger.info("🔌 MCP call failed - could fall back to local tool")

        # Emergency circuit breaker
        if self.error_count > MAX_ERROR_COUNT:
            logger.critical("🚨 Circuit breaker activated - too many errors!")
            # In real implementation, you might disable the system


class ContentFilterCallback(BeforeInvokeCallback[CallbackContext, bool]):
    """Example content filtering for AI inputs."""

    async def execute(self, context: CallbackContext) -> bool:
        """Filter inappropriate content before AI processing."""
        if context.invocation_type != InvocationType.AI:
            return True  # Only filter AI calls

        if context.metadata and "input" in context.metadata:
            input_text = str(context.metadata["input"]).lower()

            # Block inappropriate content
            blocked_words = ["spam", "inappropriate", "banned"]
            if any(word in input_text for word in blocked_words):
                logger.warning("🚫 Content filter blocked AI call with inappropriate content")
                return False

        return True


async def example_ai_function(state: AgentState) -> AgentState:
    """Example AI function that will trigger callbacks."""
    # Simulate AI processing
    response = Message.text_message("This is an AI response based on the input.", role="assistant")
    state.context.append(response)
    return state


async def example_tool_function(query: str) -> str:
    """Example tool function."""
    return f"Tool processed: {query}"


def setup_callback_manager() -> CallbackManager:
    """Set up a callback manager with validation callbacks."""
    callback_manager = CallbackManager()

    # Security validation (before calls)
    security_callback = SecurityValidationCallback()
    callback_manager.add_before_invoke_callback(security_callback)

    # Content filtering (before AI calls)
    content_filter = ContentFilterCallback()
    callback_manager.add_before_invoke_callback(content_filter)

    # Audit logging (after successful calls)
    audit_callback = AuditLoggingCallback()
    callback_manager.add_after_invoke_callback(audit_callback)

    # Error handling (on failures)
    error_callback = ErrorHandlingCallback()
    callback_manager.add_error_callback(error_callback)

    return callback_manager


async def run_example():
    """Run example showing callback validation in action."""
    logger.info("🚀 Starting PyAgenity callback validation example")

    # Set up callback manager with validation logic
    callback_manager = setup_callback_manager()

    # Create state graph with callback manager
    graph = StateGraph()

    # Add nodes
    graph.add_node("ai_node", example_ai_function)
    graph.add_node("tool_node", example_tool_function)

    # Set entry point
    graph.set_entry_point("ai_node")

    # Add edges
    graph.add_edge("ai_node", "tool_node")

    # Compile with callback manager
    compiled = graph.compile(callback_manager=callback_manager)

    # Create initial state
    initial_state = AgentState(context=[Message.text_message("Hello, how can you help me today?")])

    logger.info("📊 Running graph with callback validation...")

    try:
        # Run the graph - this will trigger callbacks
        result = await compiled.ainvoke(initial_state)
        logger.info("✅ Graph execution completed successfully")
        logger.info(f"📈 Final state has {len(result.context)} messages")

    except Exception as e:
        logger.error(f"❌ Graph execution failed: {e}")

    # Show audit log
    audit_callback = None
    for callback in callback_manager._after_invoke_callbacks:
        if isinstance(callback, AuditLoggingCallback):
            audit_callback = callback
            break

    if audit_callback:
        logger.info(f"📋 Audit Log ({len(audit_callback.get_audit_log())} entries):")
        for entry in audit_callback.get_audit_log():
            logger.info(
                f"  - {entry['timestamp']}: {entry['invocation_type']} call to {entry['function_name']}"
            )


async def example_blocked_calls():
    """Example showing how security validation blocks dangerous calls."""
    logger.info("🛡️  Testing security validation with blocked calls")

    callback_manager = setup_callback_manager()

    # Create test contexts that should be blocked
    dangerous_contexts = [
        CallbackContext(
            invocation_type=InvocationType.TOOL,
            node_name="dangerous_node",
            function_name="delete_all",
            metadata={},
        ),
        CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="ai_node",
            function_name="process_input",
            metadata={"input": "What's my password for the secret database?"},
        ),
    ]

    for context in dangerous_contexts:
        logger.info(f"Testing: {context.function_name}")
        allowed = await callback_manager.execute_before_invoke_callbacks(context)
        logger.info(f"Result: {'✅ Allowed' if allowed else '❌ Blocked'}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(run_example())
    asyncio.run(example_blocked_calls())

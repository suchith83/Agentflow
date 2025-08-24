import logging
from typing import TypeVar

from pyagenity.utils import Message

from .agent_state import AgentState
from .base_context import BaseContextManager


S = TypeVar("S", bound=AgentState)

logger = logging.getLogger(__name__)


class MessageContextManager(BaseContextManager[S]):
    """
    Manages the context field for AI interactions.

    This class trims the context (message history) based on a maximum number of user messages,
    ensuring the first message (usually a system prompt) is always preserved.
    Generic over AgentState or its subclasses.
    """

    def __init__(self, max_messages: int = 10) -> None:
        """
        Initialize the MessageContextManager.

        Args:
            max_messages (int): Maximum number of
                user messages to keep in context. Default is 10.
        """
        self.max_messages = max_messages
        logger.debug("Initialized MessageContextManager with max_messages=%d", max_messages)

    def _trim(self, messages: list[Message]) -> list[Message] | None:
        """
        Trim messages keeping system messages and most recent user messages.

        Returns None if no trimming is needed, otherwise returns the trimmed list.
        """
        # check context is empty
        if not messages:
            logger.debug("No messages to trim; context is empty")
            return None

        # Count user messages
        user_message_count = sum(1 for msg in messages if msg.role == "user")

        if user_message_count <= self.max_messages:
            # no trimming needed
            logger.debug(
                "No trimming needed; context is within limits (%d user messages)",
                user_message_count,
            )
            return None

        # Separate system messages (usually at the beginning)
        system_messages = [msg for msg in messages if msg.role == "system"]
        non_system_messages = [msg for msg in messages if msg.role != "system"]

        # Keep only the most recent messages that include max_messages user messages
        final_non_system = []
        user_count = 0

        # Iterate from the end to keep most recent messages
        for msg in reversed(non_system_messages):
            if msg.role == "user":
                if user_count >= self.max_messages:
                    break
                user_count += 1
            final_non_system.insert(0, msg)  # Insert at beginning to maintain order

        # Combine system messages (at start) with trimmed conversation
        trimmed_messages = system_messages + final_non_system

        logger.debug(
            "Trimmed from %d to %d messages (%d user messages kept)",
            len(messages),
            len(trimmed_messages),
            user_count,
        )

        return trimmed_messages

    def trim_context(self, state: S) -> S:
        """
        Trim the context in the given AgentState based on the maximum number of user messages.

        The first message (typically a system prompt) is always preserved. Only the most recent
        user messages up to `max_messages` are kept, along with the first message.

        Args:
            state (AgentState): The agent state containing the context to trim.

        Returns:
            S: The updated agent state with trimmed context.
        """
        messages = state.context
        trimmed_messages = self._trim(messages)
        if trimmed_messages is not None:
            state.context = trimmed_messages
        return state

    async def atrim_context(self, state: S) -> S:
        """
        Asynchronous version of trim_context.

        Args:
            state (AgentState): The agent state containing the context to trim.

        Returns:
            S: The updated agent state with trimmed context.
        """
        messages = state.context
        trimmed_messages = self._trim(messages)
        if trimmed_messages is not None:
            state.context = trimmed_messages
        return state

# from pyagenity.graph.state.state import AgentState
# from .base_context import BaseContextManager


# class SummaryContextManager(BaseContextManager):
#     """
#     Manages the context field for AI interactions.

#     This class trims the context (message history) based on a maximum number of user messages,
#     ensuring the first message (usually a system prompt) is always preserved.
#     """

#     def __init__(self, model_name: str) -> None:
#         """
#         Initialize the SummaryContextManager.

#         Args:
#             model_name (str): The name of the model to use for summarization.
#         """
#         self.model_name = model_name
#         if model_name is None:
#             raise ValueError("Model name must be provided.")

#     def trim_context(self, state: AgentState) -> AgentState:
#         """
#         Trim the context in the given AgentState based on the maximum number of user messages.

#         The first message (typically a system prompt) is always preserved. Only the most recent
#         user messages up to `max_messages` are kept, along with the first message.

#         Args:
#             state (AgentState): The agent state containing the context to trim.

#         Returns:
#             AgentState: The updated agent state with trimmed context.
#         """
#         messages = state.context
#         # check context is empty
#         if not messages:
#             return state

#         if len(messages) <= self.max_messages:
#             # no trimming needed
#             return state

#         # Keep first message (usually system prompt)
#         # and recent messages
#         first_message = messages[0]
#         # now keep last messages from user set values
#         # but we have to count from the user message
#         final_messages = []
#         user_message_count = 0

#         for i in range(len(messages)):
#             if messages[i].role == "user":
#                 user_message_count += 1

#             if user_message_count > self.max_messages:
#                 break

#             final_messages.append(messages[i])

#         state.context = [first_message] + final_messages
#         return state

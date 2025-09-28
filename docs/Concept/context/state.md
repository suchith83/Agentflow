# State
The state in PyAgenity refers to the current context and memory of the conversation. It is managed at three levels: short term memory, conversation history, and long term memory.

## Short Term Memory
Short term memory is managed using the `AgentState` class, which holds the current conversation context. The context is a list of `Message` objects that represent the dialogue between the user and the assistant.

## Conversation History
Conversation history is managed using the `Checkpointer` class, which stores a more persistent record of past interactions. This data is not shared with the LLM and is primarily used for UI display and analytics purposes.

## Long Term Memory
Long term memory is managed using the `Store` class, which provides a more permanent storage solution for user preferences, past interactions, and other relevant data. This data is not directly shared with the LLM but can be easily attached to LLM calls using the `Store` class in PyAgenity.


# Short Term Memory:
which is agent state itself, which is pydantic model, so you can easily extend it with your own fields.
Default fields are:
- context: List[Message] = []  # The conversation history
- context_summary: Optional[str] = None  # A summary of the conversation history
- execution_meta: ExecMeta  # Metadata about the execution

Context is a list of Message objects, use invoke or stream is called, the user message will be added to the context, then the agent will process the message, and the response will be added to the context as well, Incase AI calls tools, the tool call message and tool response message will be added to the context as well.

Context Summary is a optional field, you can use it to store a summary of the conversation history, so you can use it to provide context to the LLM without sending the entire conversation history.

# Now how to trim context if it gets too long?
You can use ContextManager to manage the context length, It has one method `trim_context` which will trim the context to the specified length. You can implement your own context manager by extending the `BaseContextManager` class, and override the `trim_context` method, this methods take current state and ouput the new state, so inside the function you can do anything you want, like summarization, or just remove old messages based on token or message count.

And this context manager need to be passed when you create the StateGraph
and this will be automatically called after one graph invocation is done, so you don't need to call it manually.
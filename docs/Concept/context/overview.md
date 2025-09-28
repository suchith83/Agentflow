In PyAgenity, we can manage context at 3 levels:

1. Short term memory, which is the current conversation history stored in the AgentState. This is typically a list of Message objects that represent the dialogue between the user and the assistant, and in next iteration it will be shared with the LLM.
2. Conversation history, which is a more persistent records of past interactions. This will be stored in database, and this data will not share with the LLM, its more likely for UI to show the history to user, or for analytics purpose. This concept is called Checkpointer in PyAgenity.
3. Long term memory, which is a more permanent storage of information that can be used across multiple conversations. This can include user preferences, past interactions, and other relevant data that can help improve the user experience. This data is typically stored in a separate database and is not directly shared with the LLM, you can easily attach with a llm calls using Store in PyAgenity.

## Short Term Memory
Short term memory is managed using the `AgentState` class, which holds the current conversation context. The context is a list of `Message` objects that represent the dialogue between the user and the assistant.
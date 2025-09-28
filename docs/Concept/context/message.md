In this library Message is a core concept. A Message is Data Class that is used to communicate with ai and store in state (as short term memory) and conversation history.

A Message has the following properties:
1. `role`: The role of the message sender. It can be one of the following values: "user", "assistant", "system", or "tool".
2. `content`: The content of the message. It can be a Text Block or a more complex object depending on the role.
3. `delta`: This property is used for straming responses from the AI, it contains partial content of the message. when its true the content message is partial. And for False its complete.
4. `tool_calls`: This is used to represent calls to tools or mcp tools need to be called by the ai. It contains information about the tool being called and its arguments.
5. `timestamp`: The timestamp when the message was created.
6. `metadata`: Additional metadata associated with the message.
7. `usages`: This property is used to track token usage for the message, including prompt tokens, completion tokens, and total tokens.

Methods:
1. `text_message`: A class method to create a message from plain text.
2. `tool_message`: A class method to create a message representing a tool call.
3. `text`: A class method to extract plain text from the message content, This will return text only if the content is a Text Block, or ToolResult Block.
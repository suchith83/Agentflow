Instead of security, lets Create callback functions, so user can define their own validation logic.

Callbacks are
1. before_invoke: Called before the AI model is invoked, allowing for input validation and modification.
2. after_invoke: Called after the AI model is invoked, allowing for output validation and modification.
3. on_error: Called when an error occurs during invocation, allowing for error handling and logging.

Make this generic, and add types also,
before_invoke -> add types (AI, TOOL, MCP)

Now Can you add that in the code
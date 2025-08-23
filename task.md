Task: Lets allow user to setup event Publisher
From this library we will publish event and user can subscribe it
and receive notifications for specific events.

We will publish all events, it's up to the user what they want to subscribe to.

I created based publisher

Events
1. Node Execution
2. Internal Tool Execution
3. MCP Tool Execution
4. State Update

Graph Execution Started / Finished
Notify when the entire graph starts or completes execution.

Node Execution Started / Finished / Failed
Separate events for start, success, and failure of node execution.

Tool Execution Failed
Capture errors or exceptions during tool execution.

State Checkpoint / Restore
When state is saved (checkpointed) or restored.

Custom User Event
Allow users to emit their own custom events.

Pause / Resume Execution
When execution is paused or resumed (if supported).

Message Sent / Received
If your system involves messaging, notify on message events.

Dependency Injection Event
When a dependency is injected or resolved.

Graph Validation / Compilation
When a graph is validated or compiled.

Subscription Change
When a user subscribes or unsubscribes to an event.


# Now We need to create some pre-built publishers
1. Redis
2. Kafka

But these dependencies need to be installed separately
only if the user wants to use them,
it wont be installed by default.
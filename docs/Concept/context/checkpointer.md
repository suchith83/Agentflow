In Pyagenity Library checkpointer concept is little different from other libraries. In PyAgenity, checkpointer is used to manage the conversation history, which is the persistent records of past interactions. This data is typically stored in a database and is not directly shared with the LLM, but it can be used for UI display and analytics purposes.

By default library provide InMemoryCheckpointer which is not persistent, its just for testing purpose. For production use cases you can use PgCheckpointer.

And the checkpointer is designed in such way that database calls are minimized, and its packed with faster database to maintain state cached in memory, and only when graph invocation is done, the state will be persisted to database, otherwise it will be save in faster database like redis, so it can match the speed of LLM calls, and database calls are minimized.

## Available Checkpointers
- InMemoryCheckpointer: This is the default checkpointer which is not persistent, its just for testing purpose.
- PgCheckpointer: This checkpointer uses Postgres as the backend database, and it requires `psycopg2` package to be installed. You can install it using `pip install pyagenity[postgres]`. This checkpointer is suitable for production use cases.


## When to use Checkpointer
You should use checkpointer when you want to maintain the conversation history, and you want to use it for UI display and analytics purposes. If you don't need to maintain the conversation history, then you can use InMemoryCheckpointer.


Usually checkpointer handled internally, and dont exopose to the user, but when its deployed with `pyagenity-cli` pag in short, all the checkpointer methods can be called vi Rest API.

Checkpointer are handled internally, and when you pass the checkpointer during graph compilation, it will be registered with `InjectQ` container, and it will be automatically injected to the nodes which require it. So you can easily access the checkpointer inside your nodes, or tools functions.

```python
def get_weather(
    location: str,
    tool_call_id: str,
    state: AgentState,
    config: dict,
    checkpointer: InMemoryCheckpointer = Inject[InMemoryCheckpointer],
) -> Message:
    pass
```

In the above example, the checkpointer will be automatically injected to the `get_weather` function, and you can use it to save or retrieve the conversation history, if you want to do any custom operations.

In the same way you can access the checkpointer inside your custom nodes, or any other places where you need it.
```python
async def main_agent(
    state: AgentState,
    config: dict,
    callback: CallbackManager = Inject[CallbackManager],
    checkpointer: InMemoryCheckpointer = Inject[InMemoryCheckpointer],
):
    pass
```
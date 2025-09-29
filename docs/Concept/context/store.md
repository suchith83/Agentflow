Last Topic in Context Management is Store
================================================================

# 🏪 Store
Store is a long-term memory solution for your agents. It allows you to save and retrieve information that can be used across multiple conversations, enhancing the agent's ability to provide relevant and personalized responses.

## When to Use Store
- You want to maintain user preferences or settings across sessions.
- You need to store information that can be referenced in future interactions.
- You want to enhance the agent's responses with historical data.
- You are building a more complex application that requires persistent data storage.


## When Not to Use Store
- Your application is simple and does not require long-term memory.
- You are only interested in the current conversation context without the need for historical data.
- You want to avoid the complexity of managing a database or external storage.


When the graph is deployed with `pyagenity-cli` plugin, all the store methods can be called via Rest API.

Internally store is not called in the process, but it can be accessed through the `Inject` container, allowing you to use it in your agent's code.

```python
async def main_agent(
    state: AgentState,
    config: dict,
    store: QdrantStore = Inject[QdrantStore],
):
    pass
```
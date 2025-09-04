Please replace InjectToolCallID and InjectState with the actual di frameworks injectq.

InjectQ is a Python library for dependency injection.
Library Code: https://github.com/Iamsdt/injectq
Documentation: https://iamsdt.github.io/injectq/

```python

# OLD Code

def get_weather(
    location: str,
    tool_call_id: InjectToolCallID,
    state: InjectState,
) -> Message:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    # You can access injected parameters here
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore

    res = f"The weather in {location} is sunny. raw"
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,  # type: ignore
    )

# NEW CODE will be using that library
    @inject
    def get_weather(
        location: str,
        tool_call_id: str,
        state: AgentState,
    ) -> Message:
        """
        Get the current weather for a specific location.
        This demo shows injectable parameters: tool_call_id and state are automatically injected.
        """
        # You can access injected parameters here
        if tool_call_id:
            print(f"Tool call ID: {tool_call_id}")
        if state and hasattr(state, "context"):
            print(f"Number of messages in context: {len(state.context)}")  # type: ignore

        res = f"The weather in {location} is sunny. raw"
        return Message.tool_message(
            content=res,
            tool_call_id=tool_call_id,  # type: ignore


# But there are 2 big challenges:
1. tool_call_id and state are mutable and change with every call. So we cannot use singleton scope. Check available scope supported by the library (https://iamsdt.github.io/injectq/scopes/understanding-scopes/#what-are-scopes).And We need to make sure that the injection happens at runtime, not at import time.
2. These fields need to be ignored when we are using all_tools in ToolNode class, this is mainly preparing the tool signature for the LLM. So we need to make sure that these fields are not included in the signature generation.


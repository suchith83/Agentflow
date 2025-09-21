Goal:
I built this library and framework, this framework was initially built on the top of
litellm, its very loosely coupled,
If the main agent function return
#Message and #Streaming

When its required then it will work,

See current implementation
async def main_agent(
    state: AgentState,
):
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    messages = convert_messages(
        system_prompts=[
            {
                "role": "system",
                "content": prompts,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "3600s",  # 👈 Cache for 1 hour
                },
            },
            {"role": "user", "content": "Today Date is 2024-06-15"},
        ],
        state=state,
    )

    mcp_tools = []

    # Check if the last message is a tool result - if so, make final response without tools
    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        # Make final response without tools since we just got tool results
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        # Regular response with tools available
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools + mcp_tools,
        )

    return response

Instead of returning response, I want to write a converter for popular library like openai and litellm and others

say function name BaseConverter, so user can return
like this

# Convert the response using the universal converter
    converter = UniversalConverter()
    converted_response = await converter.convert(
        response,
        converter_type,
        stream=stream,
        model="gemini/gemini-2.5-flash",
        **kwargs
    )


Now lets implement this, at least for openai and litellm
And convert these as optional dependencies

Remove them from
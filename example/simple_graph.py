from typing import Any

from dotenv import load_dotenv
from litellm import completion

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Message, convert_messages


load_dotenv()


def main_agent(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
    )

    return response


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.set_entry_point("MAIN")


app = graph.compile()


# now run it

inp = {"messages": [Message.from_text("Hello, world!", role="user")]}
config = {"thread_id": "12345", "recursion_limit": 5}

res = app.invoke(inp, config=config)

print(res)

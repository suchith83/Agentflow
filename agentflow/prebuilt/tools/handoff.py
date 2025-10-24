import inspect

from agentflow.state.message import Message
from agentflow.utils.command import Command


def create_handoff_tool(
    name: str,
    description: str,
    goto: str,
    update: Message | str | None = None,
):
    create_handoff_tool.__name__ = name + "handoff_tool"
    create_handoff_tool.__doc__ = description
    return Command(
        goto=goto,
        update=update,
    )


if __name__ == "__main__":
    tool_agent2 = create_handoff_tool(
        name="agent2_",
        description="Hand off to agent 2",
        goto="agent2",
        update="You are being handed off to agent 2.",
    )

    docs = inspect.getdoc(tool_agent2)
    print(docs)

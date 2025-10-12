"""Tests for ToolNode Composio adapter integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyagenity.graph.tool_node import ToolNode
from pyagenity.state import AgentState, Message
from pyagenity.utils import CallbackManager


class DummyComposioAdapter:
    def __init__(self, tools: list[dict] | None = None, exec_response: dict | None = None):
        self._tools = tools or [
            {
                "type": "function",
                "function": {
                    "name": "COMP_HELLO",
                    "description": "Say hello via Composio",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        self._exec_response = exec_response or {
            "successful": True,
            "data": {"message": "hello"},
            "error": None,
        }

    def list_raw_tools_for_llm(self, *_, **__):
        return self._tools

    def execute(self, *, slug: str, arguments: dict, user_id=None, connected_account_id=None, **_):
        return self._exec_response


@pytest.mark.asyncio
async def test_all_tools_includes_composio_tools():
    adapter = DummyComposioAdapter()
    node = ToolNode([], composio_adapter=adapter)

    tools = await node.all_tools()
    names = [t["function"]["name"] for t in tools if t.get("type") == "function"]
    assert "COMP_HELLO" in names
    assert "COMP_HELLO" in node.composio_tools


@pytest.mark.asyncio
async def test_invoke_routes_to_composio():
    adapter = DummyComposioAdapter()
    node = ToolNode([], composio_adapter=adapter)
    # Simulate discovery already populated
    node.composio_tools = ["COMP_HELLO"]

    state = AgentState()
    cfg: dict = {}
    cb = MagicMock(spec=CallbackManager)
    cb.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
    cb.execute_after_invoke = AsyncMock(side_effect=lambda ctx, input_data, result: result)
    cb.execute_on_error = AsyncMock(return_value=None)

    res = await node.invoke(
        name="COMP_HELLO",
        args={"x": 1},
        tool_call_id="tc1",
        config=cfg,
        state=state,
        callback_manager=cb,
    )

    assert isinstance(res, Message)
    assert "hello" in res.content[0].output[0]["message"]

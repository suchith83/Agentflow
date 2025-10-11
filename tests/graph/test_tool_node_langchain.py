"""Tests for ToolNode LangChain adapter integration (with dummy adapter)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyagenity.graph.tool_node import ToolNode
from pyagenity.state import AgentState
from pyagenity.utils import CallbackManager
from pyagenity.state.message import Message


class DummyLangChainAdapter:
    def __init__(self, tools: list[dict] | None = None, exec_response: dict | None = None):
        self._tools = tools or [
            {
                "type": "function",
                "function": {
                    "name": "lc_search",
                    "description": "Search via LangChain",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        self._exec_response = exec_response or {
            "successful": True,
            "data": {"message": "ok"},
            "error": None,
        }

    def list_tools_for_llm(self, *_, **__):
        return self._tools

    def execute(self, *, name: str, arguments: dict):  # mimic the adapter API
        return self._exec_response


@pytest.mark.asyncio
async def test_all_tools_includes_langchain_tools():
    adapter = DummyLangChainAdapter()
    node = ToolNode([], langchain_adapter=adapter)

    tools = await node.all_tools()
    names = [t["function"]["name"] for t in tools if t.get("type") == "function"]
    assert "lc_search" in names
    assert "lc_search" in node.langchain_tools


@pytest.mark.asyncio
async def test_invoke_routes_to_langchain():
    adapter = DummyLangChainAdapter()
    node = ToolNode([], langchain_adapter=adapter)
    # Simulate discovery already populated
    node.langchain_tools = ["lc_search"]

    state = AgentState()
    cfg: dict = {}
    cb = MagicMock(spec=CallbackManager)
    cb.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
    cb.execute_after_invoke = AsyncMock(side_effect=lambda ctx, input_data, result: result)
    cb.execute_on_error = AsyncMock(return_value=None)

    res = await node.invoke(
        name="lc_search",
        args={"q": "test"},
        tool_call_id="lc1",
        config=cfg,
        state=state,
        callback_manager=cb,
    )

    assert isinstance(res, Message)
    assert res.content is not None

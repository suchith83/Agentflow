"""Tests for agentflow.core.llm.caller.call_llm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.core.llm.caller import (
    _extract_responses_text,
    call_llm,
)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_DETECT = "agentflow.core.llm.caller.detect_provider"
_CREATE = "agentflow.core.llm.caller.create_llm_client"
_CALL_GOOGLE = "agentflow.core.llm.caller._call_google"
_CALL_RESP = "agentflow.core.llm.caller._call_openai_responses"
_CALL_CHAT = "agentflow.core.llm.caller._call_openai_chat"

_DUMMY = ("text", 10, 5, 0)


@pytest.mark.anyio
async def test_google_model_dispatches_to_google():
    with (
        patch(_DETECT, return_value="google"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gemini-2.0-flash", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_default_dispatches_to_responses():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_chat_style_dispatches_to_chat():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello", api_style="chat")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_responses_style_explicit():
    """Explicitly passing api_style='responses' still hits the Responses path."""
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock_resp,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gpt-4o-mini", "hello", api_style="responses")

    mock_resp.assert_called_once()
    mock_chat.assert_not_called()


@pytest.mark.anyio
async def test_api_style_irrelevant_for_google():
    """api_style has no effect when the provider is Google."""
    with (
        patch(_DETECT, return_value="google"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock_google,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gemini-2.0-flash", "hello", api_style="chat")

    mock_google.assert_called_once()
    mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Parameters forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_system_prompt_forwarded_to_responses():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", json_mode=True)

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."
    assert kwargs["json_mode"] is True


@pytest.mark.anyio
async def test_system_prompt_forwarded_to_chat():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", api_style="chat")

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."


# ---------------------------------------------------------------------------
# _extract_responses_text
# ---------------------------------------------------------------------------

def _make_response(output_text=None, output=None):
    r = MagicMock()
    r.output_text = output_text
    r.output = output or []
    return r


def test_extract_uses_output_text_property():
    r = _make_response(output_text="  hello  ")
    assert _extract_responses_text(r) == "hello"


def test_extract_falls_back_to_output_items():
    part = MagicMock()
    part.type = "output_text"
    part.text = "fallback"

    item = MagicMock()
    item.type = "message"
    item.content = [part]

    r = _make_response(output_text=None, output=[item])
    assert _extract_responses_text(r) == "fallback"


def test_extract_returns_empty_when_no_text():
    r = _make_response(output_text=None, output=[])
    assert _extract_responses_text(r) == ""

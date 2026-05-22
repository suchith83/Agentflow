"""Shared single-turn LLM call utility.

Centralises the Google / OpenAI dispatch that was duplicated across
SummaryContextManager, LLMCallerMixin (eval judge), and UserSimulator.

Returns a plain 4-tuple so callers can choose how much they consume:

    text, *_ = await call_llm(...)              # only text
    text, inp, out, cache = await call_llm(...)  # text + token counts

The Agent class keeps its own execution path (streaming, tools, retry, etc.)
and is unaffected by this module.
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.core.llm.client_factory import create_llm_client, detect_provider


logger = logging.getLogger("agentflow.llm.caller")


async def call_llm(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    json_mode: bool = False,
    use_vertex_ai: bool = False,
) -> tuple[str, int, int, int]:
    """Single-turn LLM call with provider auto-detection.

    Args:
        model: Model identifier (e.g. ``"gemini-2.0-flash"``, ``"gpt-4o-mini"``).
            Provider is inferred from the name via ``detect_provider``.
        prompt: The user-turn content to send.
        system_prompt: Optional system instruction prepended to the request.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        json_mode: When ``True``, instructs the provider to return valid JSON.
        use_vertex_ai: When ``True``, force Google Vertex AI client.

    Returns:
        ``(text, input_tokens, output_tokens, cache_read_tokens)`` — plain tuple.
        Token counts are 0 when the provider does not report them.
    """
    provider = detect_provider(model, use_vertex_ai=use_vertex_ai)
    client = create_llm_client(provider, use_vertex_ai=use_vertex_ai)

    if provider == "google":
        return await _call_google(
            client, model, prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    return await _call_openai(
        client, model, prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _call_google(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> tuple[str, int, int, int]:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = (response.text or "").strip()
    inp = out = cache = 0
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        inp = getattr(meta, "prompt_token_count", 0) or 0
        out = getattr(meta, "candidates_token_count", 0) or 0
        cache = getattr(meta, "cached_content_token_count", 0) or 0

    return text, inp, out, cache


async def _call_openai(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> tuple[str, int, int, int]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    text = (response.choices[0].message.content or "").strip()
    inp = out = cache = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cache = getattr(details, "cached_tokens", 0) or 0

    return text, inp, out, cache

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from pyagenity.utils.message import Message, TokenUsages
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

from .base_converter import BaseConverter


logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("model_dump() failed on %s: %s", type(obj), exc)
    # Best-effort fallback
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {"value": str(obj)}


class OpenAIConverter(BaseConverter):
    async def convert_response(self, response: Any, **kwargs) -> Message:
        data = _as_dict(response)

        # Detect Responses API vs Chat Completions
        if "choices" in data or data.get("object") in {"chat.completion", "chat.completions"}:
            return self._from_chat_completion(data)
        if "output" in data or data.get("object") == "response":
            return self._from_responses_api(data)

        # Fallback: pull text-ish fields
        text = (
            data.get("content")
            or data.get("text")
            or data.get("message", {}).get("content", "")
            or ""
        )
        msg = Message.create(role="assistant", content=str(text))
        msg.raw = data
        msg.metadata.update(
            {
                "provider": "openai",
                "object": data.get("object", ""),
            }
        )
        return msg

    def _extract_usages_from_chat(self, data: dict[str, Any]) -> TokenUsages | None:
        usage = data.get("usage") or {}
        if not usage:
            return None
        return TokenUsages(
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=(usage.get("prompt_tokens_details", {}) or {}).get(
                "reasoning_tokens", 0
            ),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )

    def _extract_usages_from_response(self, data: dict[str, Any]) -> TokenUsages | None:
        usage = data.get("usage") or {}
        if not usage:
            return None
        # Responses API typically: {input_tokens, output_tokens, total_tokens, reasoning_tokens?}
        return TokenUsages(
            completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )

    def _from_chat_completion(self, data: dict[str, Any]) -> Message:
        choices = data.get("choices") or []
        msg_dict: dict[str, Any] = choices[0].get("message", {}) if choices else {}

        # content can be string or array of parts
        content = msg_dict.get("content", "")
        if isinstance(content, list):
            # flatten text parts if present
            content = "".join([p.get("text", "") for p in content if isinstance(p, dict)])

        tool_calls = msg_dict.get("tool_calls") or []
        reasoning = (
            msg_dict.get("reasoning")
            or msg_dict.get("reasoning_content")
            or data.get("reasoning", {}).get("summary", "")
            or ""
        )

        usages = self._extract_usages_from_chat(data)
        meta = {
            "provider": "openai",
            "model": data.get("model", ""),
            "id": data.get("id", ""),
            "object": data.get("object", ""),
            "created": data.get("created", data.get("created_at")),
            "finish_reason": (choices[0] or {}).get("finish_reason") if choices else None,
            "prompt_tokens_details": (data.get("usage", {}) or {}).get("prompt_tokens_details", {}),
            "completion_tokens_details": (data.get("usage", {}) or {}).get(
                "completion_tokens_details", {}
            ),
        }
        message = Message.create(
            role="assistant",
            content=content or "",
            reasoning=reasoning or "",
            tools_calls=tool_calls or None,
            meta={k: v for k, v in meta.items() if v is not None},
            raw=data,
        )
        message.usages = usages
        return message

    def _from_responses_api(self, data: dict[str, Any]) -> Message:
        # The Responses API has output: list[items], each with content parts
        output = data.get("output", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in output:
            # item.content: list of parts
            parts = (item or {}).get("content", [])
            for part in parts:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type") or part.get("content_type")
                if ptype in {"output_text", "text"}:
                    text_parts.append(part.get("text", ""))
                elif ptype in {"tool_call", "function_call"}:
                    # Preserve full tool call in tool_calls
                    tool_calls.append(part)
                else:
                    # Non-text outputs (audio/image/video/etc.) are preserved in raw
                    pass

        usages = self._extract_usages_from_response(data)
        meta = {
            "provider": "openai",
            "model": data.get("model", ""),
            "id": data.get("id", ""),
            "object": data.get("object", ""),
            "created": data.get("created", data.get("created_at")),
            "status": data.get("status"),
        }
        message = Message.create(
            role="assistant",
            content="".join(text_parts),
            tools_calls=tool_calls or None,
            meta={k: v for k, v in meta.items() if v is not None},
            raw=data,
        )
        message.usages = usages
        return message

    def convert_streaming_response(
        self, response: Any, **kwargs
    ) -> AsyncGenerator[EventModel | Message, None]:
        base_config = kwargs.get("config", {})
        node_name = kwargs.get("node_name", "")

        handler = _OpenAIStreamNormalizer(
            base_config=base_config,
            node_name=node_name,
            converter=self,
            response=response,
        )
        return handler.run()


class _OpenAIStreamNormalizer:
    def __init__(
        self,
        base_config: dict,
        node_name: str,
        converter: OpenAIConverter,
        response: Any,
    ) -> None:
        self.base_config = base_config
        self.node_name = node_name
        self.converter = converter
        self.response = response

        self.stream_event = EventModel.stream(base_config, node_name=node_name)
        self.accumulated_text = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_ids: set[str] = set()
        self.seq = 0
        self.final_raw: dict[str, Any] | None = None

    async def aiter(self, obj):
        if hasattr(obj, "__aiter__"):
            async for x in obj:
                yield x
        else:
            for x in obj:
                yield x

    def yield_text_delta(self, txt: str) -> EventModel | None:
        if not txt:
            return None
        self.seq += 1
        self.stream_event.sequence_id = self.seq
        self.stream_event.content = txt
        self.stream_event.data = {}
        self.stream_event.content_type = [ContentType.TEXT]
        return self.stream_event

    def yield_update(self, data: dict[str, Any]) -> EventModel:
        self.seq += 1
        self.stream_event.sequence_id = self.seq
        self.stream_event.content = ""
        self.stream_event.data = data
        self.stream_event.content_type = [ContentType.UPDATE]
        return self.stream_event

    async def handle_chat_chunk(self, dchunk: dict[str, Any]) -> EventModel | None:
        choices = dchunk.get("choices", [])
        delta = choices[0].get("delta") if choices else None
        if not delta:
            return None
        if isinstance(delta, dict):
            txt = delta.get("content")
            if txt:
                self.accumulated_text += txt
                return self.yield_text_delta(txt)
            dtools = delta.get("tool_calls")
        else:
            txt = getattr(delta, "content", None)
            if txt:
                self.accumulated_text += txt
                return self.yield_text_delta(txt)
            dtools = getattr(delta, "tool_calls", None)
        if dtools:
            for tc in dtools:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id and tc_id not in self.tool_ids:
                    self.tool_ids.add(tc_id)
                    self.tool_calls.append(_as_dict(tc))
            return self.yield_update({"tool_calls": self.tool_calls})
        return None

    def handle_responses_event(self, dchunk: dict[str, Any]) -> EventModel | str | None:
        etype = dchunk.get("type")
        if not etype:
            return None
        if etype in {"response.text.delta", "response.output_text.delta"}:
            delta_text = dchunk.get("delta") or ""
            if delta_text:
                self.accumulated_text += delta_text
                meta_keys = ("item_id", "output_index", "content_index")
                meta = {k: dchunk.get(k) for k in meta_keys if dchunk.get(k) is not None}
                ev = self.yield_text_delta(delta_text)
                if ev:
                    ev.data = {"event_type": etype, **meta}
                    return ev
                return None
        if etype == "response.content_part.added":
            part = dchunk.get("part") or {}
            ptype = part.get("type") or part.get("content_type")
            if ptype in {"output_text", "text"} and part.get("text"):
                txt = part.get("text", "")
                self.accumulated_text += txt
                ev = self.yield_text_delta(txt)
                if ev:
                    ev.data = {"event_type": etype, "part": part}
                    return ev
            else:
                return self.yield_update({"event_type": etype, "part": part})
        if etype in {
            "response.audio.delta",
            "response.output_audio.delta",
            "response.audio_transcript.delta",
        }:
            meta_keys = ("item_id", "output_index", "content_index")
            meta = {k: dchunk.get(k) for k in meta_keys if dchunk.get(k) is not None}
            return self.yield_update(
                {
                    "event_type": etype,
                    "delta": dchunk.get("delta"),
                    **meta,
                }
            )
        if etype in {
            "response.completed",
            "response.completed.success",
            "response.completed_with_usage",
        }:
            self.final_raw = dchunk.get("response") or dchunk
            return "__completed__"
        if etype in {"response.error", "error"}:
            ev = self.yield_update({"event_type": etype, "error": dchunk.get("error", dchunk)})
            if ev:
                ev.is_error = True
                ev.content_type = [ContentType.ERROR]
                return ev
        return None

    async def run(self) -> AsyncGenerator[EventModel | Message, None]:
        async for chunk in self.aiter(self.response):
            dchunk = _as_dict(chunk)
            if "choices" in dchunk:
                ev = await self.handle_chat_chunk(dchunk)
                if ev:
                    yield ev
                continue
            ev2 = self.handle_responses_event(dchunk)
            if ev2 == "__completed__":
                break
            if ev2:
                yield ev2
            elif dchunk.get("object") in {"response", "chat.completion"}:
                self.final_raw = dchunk

        # Finalize
        final_data = _as_dict(self.final_raw) if self.final_raw else {}
        if final_data:
            if "choices" in final_data or final_data.get("object") in {
                "chat.completion",
                "chat.completions",
            }:
                final_msg = self.converter._from_chat_completion(final_data)
            else:
                final_msg = self.converter._from_responses_api(final_data)
        else:
            final_msg = Message.create(role="assistant", content=self.accumulated_text)
            final_msg.metadata.update({"provider": "openai"})

        end_event = EventModel.default(
            self.base_config,
            data={
                "final_response": final_msg.content or self.accumulated_text,
                "messages": [final_msg.model_dump()],
                "tool_calls": final_msg.tools_calls or self.tool_calls or [],
            },
            content_type=[ContentType.MESSAGE, ContentType.TEXT],
            event=Event.STREAMING,
            event_type=EventType.END,
            node_name=self.node_name,
        )
        yield end_event
        yield final_msg

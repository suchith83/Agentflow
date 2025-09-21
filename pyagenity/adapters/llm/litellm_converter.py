from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, cast

from pyagenity.utils.message import Message, TokenUsages, generate_id
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

from .base_converter import BaseConverter


logger = logging.getLogger(__name__)


try:
    from litellm import CustomStreamWrapper
    from litellm.types.utils import ModelResponse, ModelResponseStream

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class LiteLLMConverter(BaseConverter):
    async def convert_response(self, response: ModelResponse) -> Message:
        if not HAS_LITELLM:
            raise ImportError("litellm is not installed. Please install it to use this converter.")

        data = response.model_dump()

        usages_data = data.get("usage", {})

        usages = TokenUsages(
            completion_tokens=usages_data.get("completion_tokens", 0),
            prompt_tokens=usages_data.get("prompt_tokens", 0),
            total_tokens=usages_data.get("total_tokens", 0),
            cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0),
            reasoning_tokens=usages_data.get("prompt_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
        )

        created_date = data.get("created", datetime.now())

        # check tools calls
        tools_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

        tool_call_id = tools_calls[0].get("id") if tools_calls else None

        logger.debug("Creating message from model response with id: %s", response.id)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            content = ""
        return Message(
            message_id=generate_id(response.id),
            role="assistant",
            content=content,
            reasoning=data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", ""),
            timestamp=created_date,
            metadata={
                "provider": "litellm",
                "model": data.get("model", ""),
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason", "UNKNOWN"),
                "object": data.get("object", ""),
                "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
                "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
            },
            usages=usages,
            raw=data,
            tools_calls=tools_calls if tools_calls else None,
            tool_call_id=tool_call_id,
        )

    def _process_chunk(
        self,
        chunk: ModelResponseStream | None,
        stream_event: EventModel,
        seq: int,
        accumulated_content: str,
        accumulated_reasoning_content: str,
        tool_calls: list,
        tool_ids: set,
    ) -> tuple[str, str, list, int]:
        if not chunk:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq

        msg: ModelResponseStream = chunk  # type: ignore
        if msg is None:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq
        if msg.choices is None or len(msg.choices) == 0:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq
        delta = msg.choices[0].delta
        if delta is None:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq

        stream_event.content = delta.content if delta.content else ""
        stream_event.data = {
            "reasoning_content": getattr(delta, "reasoning_content", "") or "",
        }
        seq += 1
        stream_event.sequence_id = seq

        accumulated_content += delta.content if delta.content else ""
        accumulated_reasoning_content += getattr(delta, "reasoning_content", "") or ""
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if not tc:
                    continue

                if tc.id in tool_ids:
                    continue

                tool_ids.add(tc.id)
                tool_calls.append(tc.model_dump())

        return accumulated_content, accumulated_reasoning_content, tool_calls, seq

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: CustomStreamWrapper,
        meta: dict | None = None,
    ) -> AsyncGenerator[EventModel | Message]:
        accumulated_content = ""
        tool_calls = []
        tool_ids = set()
        accumulated_reasoning_content = ""
        seq = 0

        stream_event = EventModel.stream(
            config,
            node_name=node_name,
            extra={"provider": "litellm"},
        )

        is_awaitable = inspect.isawaitable(stream)

        # Process chunks
        if is_awaitable:
            stream = await stream

        async for chunk in stream:
            accumulated_content, accumulated_reasoning_content, tool_calls, seq = (
                self._process_chunk(
                    chunk,
                    stream_event,
                    seq,
                    accumulated_content,
                    accumulated_reasoning_content,
                    tool_calls,
                    tool_ids,
                )
            )
            yield stream_event

            # yield tool calls as well
            if tool_calls and len(tool_calls) > 0:
                seq += 1
                stream_event.data["tool_calls"] = tool_calls
                stream_event.content = ""
                stream_event.delta = True
                stream_event.sequence_id = seq
                yield stream_event

        # Loop done
        metadata = meta or {}
        metadata["provider"] = "litellm"
        message = Message.create(
            role="assistant",
            content=accumulated_content,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls,
            meta=metadata,
        )
        yield message

        stream_event.event_type = EventType.END
        stream_event.content = accumulated_content
        stream_event.sequence_id = seq + 1
        stream_event.delta = False
        stream_event.content_type = [
            ContentType.MESSAGE,
            ContentType.REASONING,
            ContentType.TEXT,
        ]
        stream_event.data = {
            "messages": [message.model_dump()],
            "next_node": None,
            "reasoning_content": accumulated_reasoning_content,
            "final_response": accumulated_content,
            "tool_calls": tool_calls,
        }
        yield stream_event

    async def convert_streaming_response(  # type: ignore
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[EventModel | Message, None]:
        if not HAS_LITELLM:
            raise ImportError("litellm is not installed. Please install it to use this converter.")

        if isinstance(response, CustomStreamWrapper):  # type: ignore[possibly-unbound]
            stream = cast(CustomStreamWrapper, response)
            async for event in self._handle_stream(
                config or {},
                node_name or "",
                stream,
                meta,
            ):
                yield event
            return

        # what if its not a stream, let's handle fallback
        elif isinstance(response, ModelResponse):  # type: ignore[possibly-unbound]
            message = await self.convert_response(cast(ModelResponse, response))
            yield EventModel(
                event=Event.STREAMING,
                event_type=EventType.END,
                delta=False,
                content_type=[
                    ContentType.TEXT,
                    ContentType.REASONING,
                    ContentType.MESSAGE,
                ],
                content=message.content,
                sequence_id=1,
                data={"messages": [message.model_dump()]},
            )

            yield message

        raise Exception("Unsupported response type for LiteLLMConverter")

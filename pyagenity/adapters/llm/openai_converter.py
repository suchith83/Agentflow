# from __future__ import annotations

# import inspect
# import logging
# from collections.abc import AsyncGenerator, AsyncIterable
# from datetime import datetime
# from typing import Any, cast

# from pyagenity.utils.message import (
#     Message,
#     ReasoningBlock,
#     TextBlock,
#     TokenUsages,
#     generate_id,
# )
# from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

# from .base_converter import BaseConverter


# logger = logging.getLogger(__name__)


# try:  # OpenAI Python SDK v1.x
#     from openai.types.chat.chat_completion import ChatCompletion
#     from openai.types.responses.response import Response as ResponsesResponse

#     HAS_OPENAI = True
# except Exception:  # pragma: no cover - optional dependency
#     HAS_OPENAI = False


# def _message_from_chat_completion(resp: ChatCompletion) -> Message:
#     data = resp.model_dump()
#     choices = data.get("choices", [])
#     msg = choices[0].get("message", {}) if choices else {}
#     content = msg.get("content") or ""
#     reasoning_content = msg.get("reasoning_content") or ""

#     blocks: list[Any] = []
#     if content:
#         blocks.append(TextBlock(text=content))
#     if reasoning_content:
#         blocks.append(ReasoningBlock(summary=reasoning_content))

#     usages_data = data.get("usage", {}) or {}
#     usages = TokenUsages(
#         completion_tokens=usages_data.get("completion_tokens", 0) or 0,
#         prompt_tokens=usages_data.get("prompt_tokens", 0) or 0,
#         total_tokens=usages_data.get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0) or 0,
#         cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0) or 0,
#         reasoning_tokens=(usages_data.get("prompt_tokens_details", {}) or {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     tool_calls = msg.get("tool_calls") or []
#     tool_call_id = tool_calls[0].get("id") if tool_calls else None

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content,
#         reasoning=reasoning_content,
#         timestamp=datetime.fromtimestamp(data.get("created", int(datetime.now().timestamp()))),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "finish_reason": (choices[0].get("finish_reason") if choices else None)
#             or "UNKNOWN",
#             "object": data.get("object"),
#             "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
#             "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
#         },
#         usages=usages,
#         raw=data,
#         tools_calls=tool_calls or None,
#         tool_call_id=tool_call_id,
#     )


# def _message_from_responses(resp: ResponsesResponse) -> Message:
#     data = resp.model_dump()
#     # Aggregate text from output items (Responses API)
#     output_text_parts: list[str] = []
#     reasoning_parts: list[str] = []
#     for item in data.get("output", []) or []:
#         if item.get("type") == "message":
#             for c in item.get("content", []) or []:
#                 t = c.get("type")
#                 if t == "output_text" and c.get("text"):
#                     output_text_parts.append(c["text"])
#                 elif t in ("reasoning_text", "reasoning_summary_text") and c.get("text"):
#                     reasoning_parts.append(c["text"])

#     content_text = "".join(output_text_parts)
#     reasoning_text = "".join(reasoning_parts)

#     blocks: list[Any] = []
#     if content_text:
#         blocks.append(TextBlock(text=content_text))
#     if reasoning_text:
#         blocks.append(ReasoningBlock(summary=reasoning_text))

#     usages_data = data.get("usage") or {}
#     usages = TokenUsages(
#         completion_tokens=(usages_data or {}).get("output_tokens", 0) or 0,
#         prompt_tokens=(usages_data or {}).get("input_tokens", 0) or 0,
#         total_tokens=(usages_data or {}).get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=0,
#         cache_read_input_tokens=0,
#         reasoning_tokens=(usages_data or {}).get("output_tokens_details", {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content_text,
#         reasoning=reasoning_text,
#         timestamp=datetime.fromtimestamp(
#             int(cast(float, data.get("created_at", datetime.now().timestamp())))
#         ),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "object": data.get("object"),
#             "status": data.get("status"),
#         },
#         usages=usages,
#         raw=data,
#     )


# def _is_async_iterable(obj: Any) -> bool:
#     return hasattr(obj, "__aiter__")


# def _update_stream_event(
#     stream_event: EventModel,
#     seq: int,
#     text_part: str,
#     blocks: list[Any] | None,
#     data: dict[str, Any] | None = None,
# ) -> None:
#     stream_event.sequence_id = seq
#     stream_event.content = text_part
#     stream_event.content_blocks = blocks or None
#     stream_event.data = data or {}


# def _process_tool_calls(delta: Any, tool_ids: set[str], tool_calls: list[dict]) -> bool:
#     new_tools = False
#     tcs = getattr(delta, "tool_calls", None) or []
#     for tc in tcs:
#         tc_id = getattr(tc, "id", None)
#         if not tc_id or tc_id in tool_ids:
#             continue
#         tool_ids.add(tc_id)
#         try:
#             tool_calls.append(tc.model_dump())  # type: ignore[attr-defined]
#         except Exception:
#             tool_calls.append({k: getattr(tc, k) for k in dir(tc) if not k.startswith("_")})
#         new_tools = True
#     return new_tools


# def _make_blocks(text_part: str, reasoning_part: str) -> list[Any]:
#     blocks: list[Any] = []
#     if text_part:
#         blocks.append(TextBlock(text=text_part))
#     if reasoning_part:
#         blocks.append(ReasoningBlock(summary=reasoning_part))
#     return blocks


# class OpenAIConverter(BaseConverter):
#     """Normalize OpenAI Python SDK responses to Message/EventModel.

#     Supports both Chat Completions and the newer Responses API.
#     """

#     async def convert_response(self, response: Any) -> Message:
#         # Chat Completions
#         if hasattr(response, "object") and response.object == "chat.completion":
#             return _message_from_chat_completion(cast(ChatCompletion, response))

#         # Responses API
#         if hasattr(response, "object") and response.object == "response":
#             return _message_from_responses(cast(ResponsesResponse, response))

#         # dict fallback
#         if isinstance(response, dict):
#             obj = response.get("object")
#             if obj == "chat.completion":
#                 # construct fake ChatCompletion-like
#                 class _Tmp:  # minimal shim
#                     def __init__(self, d: dict):
#                         self._d = d
#                         self.id = d.get("id")

#                     def model_dump(self):
#                         return self._d

#                 return _message_from_chat_completion(cast(ChatCompletion, _Tmp(response)))
#             if obj == "response":
#                 class _Tmp2:
#                     def __init__(self, d: dict):
#                         self._d = d
#                         self.id = d.get("id")

#                     def model_dump(self):
#                         return self._d

#                 return _message_from_responses(cast(ResponsesResponse, _Tmp2(response)))

#         raise TypeError("Unsupported OpenAI response type for conversion")

#     async def convert_streaming_response(  # type: ignore
#         self,
#         config: dict,
#         node_name: str,
#         response: Any,
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         # The OpenAI SDK returns an async generator for streams in both APIs.
#         if inspect.isawaitable(response):
#             response = await response

#         # If it's an async iterable, peek the first item to route appropriately
#         if _is_async_iterable(response):
#             async for first in response:  # type: ignore
#                 async def _regen(first_item: Any, src: Any):
#                     yield first_item
#                     async for rest in src:
#                         yield rest

#                 if HAS_OPENAI and getattr(first, "object", None) == "chat.completion.chunk":
#                     async for ev in self._handle_chat_stream(
#                         config, node_name, _regen(first, response), meta
#                     ):
#                         yield ev
#                 else:
#                     async for ev in self._handle_responses_stream(
#                         config, node_name, _regen(first, response), meta
#                     ):
#                         yield ev
#                 return

#         # Non-stream -> convert and emit END event + message
#         if hasattr(response, "object") or isinstance(response, dict):
#             message = await self.convert_response(response)
#             content_value = (
#                 message.text() if isinstance(message.content, list) else cast(str,
#               message.content)
#             )
#             end_event = EventModel(
#                 event=Event.STREAMING,
#                 event_type=EventType.END,
#                 delta=False,
#                 content_type=[ContentType.TEXT, ContentType.REASONING, ContentType.MESSAGE],
#                 content=content_value,
#                 sequence_id=1,
#                 data={"messages": [message.model_dump()]},
#             )
#             if isinstance(message.content, list):
#                 end_event.content_blocks = message.content
#             yield end_event
#             yield message
#             return

#         raise Exception("Unsupported response type for OpenAIConverter")

#     async def _handle_chat_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         tool_calls: list[dict] = []
#         tool_ids: set[str] = set()
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for chunk in stream:
#             choices = getattr(chunk, "choices", []) or []
#             delta = choices[0].delta if choices else None
#             if not delta:
#                 continue

#             text_part = getattr(delta, "content", None) or ""
#             reasoning_part = getattr(delta, "reasoning_content", None) or ""

#             blocks = _make_blocks(text_part, reasoning_part)

#             seq += 1
#             _update_stream_event(
#                 stream_event,
#                 seq,
#                 text_part,
#                 blocks,
#                 {"reasoning_content": reasoning_part},
#             )
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#             if _process_tool_calls(delta, tool_ids, tool_calls):
#                 seq += 1
#                 _update_stream_event(stream_event, seq, "", None, {"tool_calls": tool_calls})
#                 stream_event.delta = True
#                 yield stream_event

#         # Final message and END event
#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks = _make_blocks(accumulated_content, accumulated_reasoning)

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             tools_calls=tool_calls,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#             "tool_calls": tool_calls,
#         }
#         yield stream_event

#     async def _handle_responses_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for ev in stream:
#             ev_type = getattr(ev, "type", None)
#             text_part = ""
#             reasoning_part = ""
#             if ev_type in ("response.output_text.delta", "response.text.delta"):
#                 text_part = getattr(ev, "delta", "") or ""
#             elif ev_type in (
#                 "response.reasoning_text.delta",
#                 "response.reasoning_summary_text.delta",
#             ):
#                 reasoning_part = getattr(ev, "delta", "") or ""
#             else:
#                 # ignore other events
#                 continue

#             blocks = _make_blocks(text_part, reasoning_part)

#             seq += 1
#             _update_stream_event(
#                 stream_event,
#                 seq,
#                 text_part,
#                 blocks,
#                 {"reasoning_content": reasoning_part, "event_type": ev_type},
#             )
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks = _make_blocks(accumulated_content, accumulated_reasoning)

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#         }
#         yield stream_event
# from __future__ import annotations

# import inspect
# import logging
# from collections.abc import AsyncGenerator, AsyncIterable
# from datetime import datetime
# from typing import Any, cast

# from pyagenity.utils.message import (
#     Message,
#     ReasoningBlock,
#     TextBlock,
#     TokenUsages,
#     generate_id,
# )
# from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

# from .base_converter import BaseConverter


# logger = logging.getLogger(__name__)


# try:  # OpenAI Python SDK v1.x
#     from openai.types.chat.chat_completion import ChatCompletion
#     from openai.types.responses.response import Response as ResponsesResponse

#     HAS_OPENAI = True
# except Exception:  # pragma: no cover - optional dependency
#     HAS_OPENAI = False


# def _message_from_chat_completion(resp: ChatCompletion) -> Message:
#     data = resp.model_dump()
#     choices = data.get("choices", [])
#     msg = choices[0].get("message", {}) if choices else {}
#     content = msg.get("content") or ""
#     reasoning_content = msg.get("reasoning_content") or ""

#     blocks: list[Any] = []
#     if content:
#         blocks.append(TextBlock(text=content))
#     if reasoning_content:
#         blocks.append(ReasoningBlock(summary=reasoning_content))

#     usages_data = data.get("usage", {}) or {}
#     usages = TokenUsages(
#         completion_tokens=usages_data.get("completion_tokens", 0) or 0,
#         prompt_tokens=usages_data.get("prompt_tokens", 0) or 0,
#         total_tokens=usages_data.get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0) or 0,
#         cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0) or 0,
#         reasoning_tokens=(usages_data.get("prompt_tokens_details", {}) or {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     tool_calls = msg.get("tool_calls") or []
#     tool_call_id = tool_calls[0].get("id") if tool_calls else None

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content,
#         reasoning=reasoning_content,
#         timestamp=datetime.fromtimestamp(data.get("created", int(datetime.now().timestamp()))),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "finish_reason": (choices[0].get("finish_reason") if choices else None)
#             or "UNKNOWN",
#             "object": data.get("object"),
#             "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
#             "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
#         },
#         usages=usages,
#         raw=data,
#         tools_calls=tool_calls or None,
#         tool_call_id=tool_call_id,
#     )


# def _message_from_responses(resp: ResponsesResponse) -> Message:
#     data = resp.model_dump()
#     # Aggregate text from output items (Responses API)
#     output_text_parts: list[str] = []
#     reasoning_parts: list[str] = []
#     for item in data.get("output", []) or []:
#         if item.get("type") == "message":
#             for c in item.get("content", []) or []:
#                 t = c.get("type")
#                 if t == "output_text" and c.get("text"):
#                     output_text_parts.append(c["text"])
#                 elif t in ("reasoning_text", "reasoning_summary_text") and c.get("text"):
#                     reasoning_parts.append(c["text"])

#     content_text = "".join(output_text_parts)
#     reasoning_text = "".join(reasoning_parts)

#     blocks: list[Any] = []
#     if content_text:
#         blocks.append(TextBlock(text=content_text))
#     if reasoning_text:
#         blocks.append(ReasoningBlock(summary=reasoning_text))

#     usages_data = data.get("usage") or {}
#     usages = TokenUsages(
#         completion_tokens=(usages_data or {}).get("output_tokens", 0) or 0,
#         prompt_tokens=(usages_data or {}).get("input_tokens", 0) or 0,
#         total_tokens=(usages_data or {}).get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=0,
#         cache_read_input_tokens=0,
#         reasoning_tokens=(usages_data or {}).get("output_tokens_details", {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content_text,
#         reasoning=reasoning_text,
#         timestamp=datetime.fromtimestamp(
#             int(cast(float, data.get("created_at", datetime.now().timestamp())))
#         ),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks = _make_blocks(accumulated_content, accumulated_reasoning)

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#         }
#         yield stream_event
# from __future__ import annotations

# import inspect
# import logging
# from collections.abc import AsyncGenerator, AsyncIterable
# from datetime import datetime
# from typing import Any, cast

# from pyagenity.utils.message import (
#     Message,
#     ReasoningBlock,
#     TextBlock,
#     TokenUsages,
#     generate_id,
# )
# from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

# from .base_converter import BaseConverter


# logger = logging.getLogger(__name__)


# try:  # OpenAI Python SDK v1.x
#     from openai.types.chat.chat_completion import ChatCompletion
#     from openai.types.responses.response import Response as ResponsesResponse

#     HAS_OPENAI = True
# except Exception:  # pragma: no cover - optional dependency
#     HAS_OPENAI = False


# def _message_from_chat_completion(resp: ChatCompletion) -> Message:
#     data = resp.model_dump()
#     choices = data.get("choices", [])
#     msg = choices[0].get("message", {}) if choices else {}
#     content = msg.get("content") or ""
#     reasoning_content = msg.get("reasoning_content") or ""

#     blocks: list[Any] = []
#     if content:
#         blocks.append(TextBlock(text=content))
#     if reasoning_content:
#         blocks.append(ReasoningBlock(summary=reasoning_content))

#     usages_data = data.get("usage", {}) or {}
#     usages = TokenUsages(
#         completion_tokens=usages_data.get("completion_tokens", 0) or 0,
#         prompt_tokens=usages_data.get("prompt_tokens", 0) or 0,
#         total_tokens=usages_data.get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0) or 0,
#         cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0) or 0,
#         reasoning_tokens=(usages_data.get("prompt_tokens_details", {}) or {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     tool_calls = msg.get("tool_calls") or []
#     tool_call_id = tool_calls[0].get("id") if tool_calls else None

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content,
#         reasoning=reasoning_content,
#         timestamp=datetime.fromtimestamp(data.get("created", int(datetime.now().timestamp()))),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "finish_reason": (choices[0].get("finish_reason") if choices else None)
#             or "UNKNOWN",
#             "object": data.get("object"),
#             "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
#             "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
#         },
#         usages=usages,
#         raw=data,
#         tools_calls=tool_calls or None,
#         tool_call_id=tool_call_id,
#     )


# def _message_from_responses(resp: ResponsesResponse) -> Message:
#     data = resp.model_dump()
#     # Aggregate text from output items (Responses API)
#     output_text_parts: list[str] = []
#     reasoning_parts: list[str] = []
#     for item in data.get("output", []) or []:
#         if item.get("type") == "message":
#             for c in item.get("content", []) or []:
#                 t = c.get("type")
#                 if t == "output_text" and c.get("text"):
#                     output_text_parts.append(c["text"])
#                 elif t in ("reasoning_text", "reasoning_summary_text") and c.get("text"):
#                     reasoning_parts.append(c["text"])

#     content_text = "".join(output_text_parts)
#     reasoning_text = "".join(reasoning_parts)

#     blocks: list[Any] = []
#     if content_text:
#         blocks.append(TextBlock(text=content_text))
#     if reasoning_text:
#         blocks.append(ReasoningBlock(summary=reasoning_text))

#     usages_data = data.get("usage") or {}
#     usages = TokenUsages(
#         completion_tokens=(usages_data or {}).get("output_tokens", 0) or 0,
#         prompt_tokens=(usages_data or {}).get("input_tokens", 0) or 0,
#         total_tokens=(usages_data or {}).get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=0,
#         cache_read_input_tokens=0,
#         reasoning_tokens=(usages_data or {}).get("output_tokens_details", {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))),
#         role="assistant",
#         content=blocks if blocks else content_text,
#         reasoning=reasoning_text,
#         timestamp=datetime.fromtimestamp(
#             int(cast(float, data.get("created_at", datetime.now().timestamp())))
#         ),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "object": data.get("object"),
#             "status": data.get("status"),
#         },
#         usages=usages,
#         raw=data,
#     )


# def _is_async_iterable(obj: Any) -> bool:
#     return hasattr(obj, "__aiter__")


# def _update_stream_event(
#     stream_event: EventModel,
#     seq: int,
#     text_part: str,
#     blocks: list[Any] | None,
#     data: dict[str, Any] | None = None,
# ) -> None:
#     stream_event.sequence_id = seq
#     stream_event.content = text_part
#     stream_event.content_blocks = blocks or None
#     stream_event.data = data or {}


# def _process_tool_calls(delta: Any, tool_ids: set[str], tool_calls: list[dict]) -> bool:
#     new_tools = False
#     tcs = getattr(delta, "tool_calls", None) or []
#     for tc in tcs:
#         tc_id = getattr(tc, "id", None)
#         if not tc_id or tc_id in tool_ids:
#             continue
#         tool_ids.add(tc_id)
#         try:
#             tool_calls.append(tc.model_dump())  # type: ignore[attr-defined]
#         except Exception:
#             tool_calls.append({k: getattr(tc, k) for k in dir(tc) if not k.startswith("_")})
#         new_tools = True
#     return new_tools


# def _make_blocks(text_part: str, reasoning_part: str) -> list[Any]:
#     blocks: list[Any] = []
#     if text_part:
#         blocks.append(TextBlock(text=text_part))
#     if reasoning_part:
#         blocks.append(ReasoningBlock(summary=reasoning_part))
#     return blocks


# class OpenAIConverter(BaseConverter):
#     """Normalize OpenAI Python SDK responses to Message/EventModel.

#     Supports both Chat Completions and the newer Responses API.
#     """

#     async def convert_response(self, response: Any) -> Message:
#         # Chat Completions
#         if hasattr(response, "object") and response.object == "chat.completion":
#             return _message_from_chat_completion(cast(ChatCompletion, response))

#         # Responses API
#         if hasattr(response, "object") and response.object == "response":
#             return _message_from_responses(cast(ResponsesResponse, response))

#         # dict fallback
#         if isinstance(response, dict):
#             obj = response.get("object")
#             if obj == "chat.completion":
#                 # construct fake ChatCompletion-like
#                 class _Tmp:  # minimal shim
#                     def __init__(self, d: dict):
#                         self._d = d
#                         self.id = d.get("id")

#                     def model_dump(self):
#                         return self._d

#                 return _message_from_chat_completion(cast(ChatCompletion, _Tmp(response)))
#             if obj == "response":
#                 class _Tmp2:
#                     def __init__(self, d: dict):
#                         self._d = d
#                         self.id = d.get("id")

#                     def model_dump(self):
#                         return self._d

#                 return _message_from_responses(cast(ResponsesResponse, _Tmp2(response)))

#         raise TypeError("Unsupported OpenAI response type for conversion")

#     async def convert_streaming_response(  # type: ignore
#         self,
#         config: dict,
#         node_name: str,
#         response: Any,
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         # The OpenAI SDK returns an async generator for streams in both APIs.
#         if inspect.isawaitable(response):
#             response = await response

#         # If it's an async iterable, peek the first item to route appropriately
#         if _is_async_iterable(response):
#             async for first in response:  # type: ignore
#                 async def _regen(first_item: Any, src: Any):
#                     yield first_item
#                     async for rest in src:
#                         yield rest

#                 if HAS_OPENAI and getattr(first, "object", None) == "chat.completion.chunk":
#                     async for ev in self._handle_chat_stream(
#                         config, node_name, _regen(first, response), meta
#                     ):
#                         yield ev
#                 else:
#                     async for ev in self._handle_responses_stream(
#                         config, node_name, _regen(first, response), meta
#                     ):
#                         yield ev
#                 return

#         # Non-stream -> convert and emit END event + message
#         if hasattr(response, "object") or isinstance(response, dict):
#             message = await self.convert_response(response)
#             end_event = EventModel(
#                 event=Event.STREAMING,
#                 event_type=EventType.END,
#                 delta=False,
#                 content_type=[ContentType.TEXT, ContentType.REASONING, ContentType.MESSAGE],
#                 content=(
#                     message.text() if isinstance(message.content, list) else cast(str,
#                      message.content)
#                 ),
#                 sequence_id=1,
#                 data={"messages": [message.model_dump()]},
#             )
#             if isinstance(message.content, list):
#                 end_event.content_blocks = message.content
#             yield end_event
#             yield message
#             return

#         raise Exception("Unsupported response type for OpenAIConverter")

#     async def _handle_chat_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         tool_calls: list[dict] = []
#         tool_ids: set[str] = set()
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for chunk in stream:
#             choices = getattr(chunk, "choices", []) or []
#             delta = choices[0].delta if choices else None
#             if not delta:
#                 continue

#             text_part = getattr(delta, "content", None) or ""
#             reasoning_part = getattr(delta, "reasoning_content", None) or ""

#             blocks = _make_blocks(text_part, reasoning_part)

#             seq += 1
#             _update_stream_event(
#                 stream_event,
#                 seq,
#                 text_part,
#                 blocks,
#                 {"reasoning_content": reasoning_part},
#             )
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#             if _process_tool_calls(delta, tool_ids, tool_calls):
#                 seq += 1
#                 _update_stream_event(stream_event, seq, "", None, {"tool_calls": tool_calls})
#                 stream_event.delta = True
#                 yield stream_event

#         # Final message and END event
#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks = _make_blocks(accumulated_content, accumulated_reasoning)

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             tools_calls=tool_calls,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#             "tool_calls": tool_calls,
#         }
#         yield stream_event

#     async def _handle_responses_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for ev in stream:
#             ev_type = getattr(ev, "type", None)
#             text_part = ""
#             reasoning_part = ""
#             if ev_type in ("response.output_text.delta", "response.text.delta"):
#                 text_part = getattr(ev, "delta", "") or ""
#             elif ev_type in (
#                 "response.reasoning_text.delta",
#                 "response.reasoning_summary_text.delta",
#             ):
#                 reasoning_part = getattr(ev, "delta", "") or ""
#             else:
#                 # ignore other events
#                 continue

#             blocks = _make_blocks(text_part, reasoning_part)

#             seq += 1
#             _update_stream_event(
#                 stream_event,
#                 seq,
#                 text_part,
#                 blocks,
#                 {"reasoning_content": reasoning_part, "event_type": ev_type},
#             )
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks = _make_blocks(accumulated_content, accumulated_reasoning)

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#         }
#         yield stream_event

# from __future__ import annotations
# import inspect
# import logging
# from collections.abc import AsyncGenerator, AsyncIterable
# from datetime import datetime
# from typing import Any, cast

# from pyagenity.utils.message import (
#     Message,
#     ReasoningBlock,
#     TextBlock,
#     TokenUsages,
#     generate_id,
# )
# from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

# from .base_converter import BaseConverter


# logger = logging.getLogger(__name__)


# try:  # OpenAI Python SDK v1.x
#     from openai.types.chat.chat_completion import ChatCompletion
#     from openai.types.responses.response import Response as ResponsesResponse

#     HAS_OPENAI = True
# except Exception:  # pragma: no cover - optional dependency
#     HAS_OPENAI = False


# def _message_from_chat_completion(resp: ChatCompletion) -> Message:
#     data = resp.model_dump()
#     choices = data.get("choices", [])
#     msg = choices[0].get("message", {}) if choices else {}
#     content = msg.get("content") or ""
#     reasoning_content = msg.get("reasoning_content") or ""
#     blocks = []
#     if content:
#         blocks.append(TextBlock(text=content))
#     if reasoning_content:
#         blocks.append(ReasoningBlock(summary=reasoning_content))

#     usages_data = data.get("usage", {}) or {}
#     usages = TokenUsages(
#         completion_tokens=usages_data.get("completion_tokens", 0) or 0,
#         prompt_tokens=usages_data.get("prompt_tokens", 0) or 0,
#         total_tokens=usages_data.get("total_tokens", 0) or 0,
#         cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0) or 0,
#         cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0) or 0,
#         reasoning_tokens=(usages_data.get("prompt_tokens_details", {}) or {}).get(
#             "reasoning_tokens", 0
#         )
#         or 0,
#     )

#     tool_calls = msg.get("tool_calls") or []
#     tool_call_id = tool_calls[0].get("id") if tool_calls else None

#     return Message(
#         message_id=generate_id(cast(str, getattr(resp, "id", None))) ,
#         role="assistant",
#         content=blocks if blocks else content,
#         reasoning=reasoning_content,
#         timestamp=datetime.fromtimestamp(data.get("created", int(datetime.now().timestamp()))),
#         metadata={
#             "provider": "openai",
#             "model": data.get("model"),
#             "finish_reason": (choices[0].get("finish_reason") if choices else None) or "UNKNOWN",
#             "object": data.get("object"),
#             "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
#             "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
#         },
#         usages=usages,
#         raw=data,


#     async def _handle_chat_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         tool_calls: list[dict] = []
#         tool_ids: set[str] = set()
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for chunk in stream:
#             choices = getattr(chunk, "choices", []) or []
#             delta = choices[0].delta if choices else None
#             if not delta:
#                 continue

#             text_part = getattr(delta, "content", None) or ""
#             reasoning_part = getattr(delta, "reasoning_content", None) or ""

#             blocks: list[Any] = []
#             if text_part:
#                 blocks.append(TextBlock(text=text_part))
#             if reasoning_part:
#                 blocks.append(ReasoningBlock(summary=reasoning_part))

#             seq += 1
#             _update_stream_event(
#                 stream_event,
#                 seq,
#                 text_part,
#                 blocks,
#                 {"reasoning_content": reasoning_part},
#             )
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#             if _process_tool_calls(delta, tool_ids, tool_calls):
#                 seq += 1
#                 _update_stream_event(stream_event, seq, "", None, {"tool_calls": tool_calls})
#                 stream_event.delta = True
#                 yield stream_event

#         # Final message and END event
#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks: list[Any] = []
#         if accumulated_content:
#             final_blocks.append(TextBlock(text=accumulated_content))
#         if accumulated_reasoning:
#             final_blocks.append(ReasoningBlock(summary=accumulated_reasoning))

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             tools_calls=tool_calls,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#             "tool_calls": tool_calls,
#         }
#         yield stream_event

#     async def _handle_responses_stream(
#         self,
#         config: dict,
#         node_name: str,
#         stream: AsyncIterable[Any],
#         meta: dict | None = None,
#     ) -> AsyncGenerator[EventModel | Message, None]:
#         accumulated_content = ""
#         accumulated_reasoning = ""
#         seq = 0

#         stream_event = EventModel.stream(
#             config,
#             node_name=node_name,
#             extra={"provider": "openai"},
#         )

#         async for ev in stream:
#             ev_type = getattr(ev, "type", None)
#             text_part = ""
#             reasoning_part = ""
#             if ev_type in ("response.output_text.delta", "response.text.delta"):
#                 text_part = getattr(ev, "delta", "") or ""
#             elif ev_type in (
#                 "response.reasoning_text.delta",
#                 "response.reasoning_summary_text.delta",
#             ):
#                 reasoning_part = getattr(ev, "delta", "") or ""
#             else:
#                 # ignore other events
#                 continue

#             blocks: list[Any] = []
#             if text_part:
#                 blocks.append(TextBlock(text=text_part))
#             if reasoning_part:
#                 blocks.append(ReasoningBlock(summary=reasoning_part))

#             seq += 1
#             stream_event.sequence_id = seq
#             stream_event.content = text_part
#             stream_event.content_blocks = blocks or None
#             stream_event.data = {"reasoning_content": reasoning_part, "event_type": ev_type}
#             yield stream_event

#             accumulated_content += text_part
#             accumulated_reasoning += reasoning_part

#         metadata = meta.copy() if meta else {}
#         metadata["provider"] = "openai"

#         final_blocks: list[Any] = []
#         if accumulated_content:
#             final_blocks.append(TextBlock(text=accumulated_content))
#         if accumulated_reasoning:
#             final_blocks.append(ReasoningBlock(summary=accumulated_reasoning))

#         message = Message.create(
#             role="assistant",
#             content=final_blocks if final_blocks else accumulated_content,
#             reasoning=accumulated_reasoning,
#             meta=metadata,
#         )
#         yield message

#         stream_event.event_type = EventType.END
#         stream_event.delta = False
#         stream_event.content = accumulated_content
#         stream_event.sequence_id = seq + 1
#         stream_event.content_blocks = final_blocks or None
#         stream_event.content_type = [
#             ContentType.MESSAGE,
#             ContentType.REASONING,
#             ContentType.TEXT,
#         ]
#         stream_event.data = {
#             "messages": [message.model_dump()],
#             "final_response": accumulated_content,
#             "reasoning_content": accumulated_reasoning,
#         }
#         yield stream_event



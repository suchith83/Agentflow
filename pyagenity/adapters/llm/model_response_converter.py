from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Callable
from typing import Any

from pyagenity.utils.message import Message

from .base_converter import BaseConverter


class ModelResponseConverter:
    """Wrap an LLM SDK call and normalize its output via a converter.

    Supports sync/async invocation and streaming. Use `invoke()` for
    non-streaming calls and `stream()` for streaming calls.
    """

    def __init__(
        self,
        response: Any | Callable[..., Any],
        converter: BaseConverter | str,
    ) -> None:
        self.response = response

        if isinstance(converter, str) and converter == "litellm":
            from .litellm_converter import LiteLLMConverter

            self.converter = LiteLLMConverter()
        else:
            raise ValueError(f"Unsupported converter: {converter}")

    async def invoke(self) -> Message:
        """Call the underlying function and convert a non-streaming response to Message."""
        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        return await self.converter.convert_response(response)  # type: ignore

    async def stream(
        self,
        config: dict,
        node_name: str,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Call the underlying function and yield normalized streaming events and final Message.

        If the function returns a non-iterable (i.e., not a stream), we fall back to
        converting it as a non-streaming response and emit a terminal END event followed
        by the final Message for consistency with streaming consumers.
        """

        if not config:
            raise ValueError("Config must be provided for streaming conversion")

        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        async for item in self.converter.convert_streaming_response(  # type: ignore
            config,
            node_name=node_name,
            response=response,
            meta=meta,
        ):
            yield item

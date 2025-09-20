"""Slim ToolNode composed from focused mixins.

This preserves the public behavior while making the implementation modular.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import typing as t

from injectq import Inject

from pyagenity.adapters.tools import ComposioAdapter
from pyagenity.graph.utils.utils import publish_event
from pyagenity.state import AgentState
from pyagenity.utils import CallbackManager
from pyagenity.utils.message import Message
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType

from . import deps
from .executors import ComposioMixin, KwargsResolverMixin, LangChainMixin, LocalExecMixin, MCPMixin
from .schema import SchemaMixin


logger = logging.getLogger(__name__)


class ToolNode(
    SchemaMixin,
    LocalExecMixin,
    MCPMixin,
    ComposioMixin,
    LangChainMixin,
    KwargsResolverMixin,
):
    """Registry for callables that exposes function specs and executes them."""

    def __init__(
        self,
        functions: t.Iterable[t.Callable],
        client: t.Any | None = None,
        composio_adapter: ComposioAdapter | None = None,
        langchain_adapter: t.Any | None = None,
    ) -> None:
        logger.info("Initializing ToolNode with %d functions", len(list(functions)))

        if client is not None:
            # Read flags dynamically so tests can patch pyagenity.graph.tool_node.HAS_*
            mod = sys.modules.get("pyagenity.graph.tool_node")
            has_fastmcp = getattr(mod, "HAS_FASTMCP", deps.HAS_FASTMCP) if mod else deps.HAS_FASTMCP
            has_mcp = getattr(mod, "HAS_MCP", deps.HAS_MCP) if mod else deps.HAS_MCP

            if not has_fastmcp or not has_mcp:
                raise ImportError(
                    "MCP client functionality requires 'fastmcp' and 'mcp' packages. "
                    "Install with: pip install pyagenity[mcp]"
                )
            logger.debug("ToolNode initialized with MCP client")

        self._funcs: dict[str, t.Callable] = {}
        self._client: t.Any | None = client
        self._composio: ComposioAdapter | None = composio_adapter
        self._langchain: t.Any | None = langchain_adapter

        for fn in functions:
            if not callable(fn):
                raise TypeError("ToolNode only accepts callables")
            self._funcs[fn.__name__] = fn

        self.mcp_tools: list[str] = []
        self.composio_tools: list[str] = []
        self.langchain_tools: list[str] = []

    async def _all_tools_async(self) -> list[dict]:
        tools: list[dict] = self.get_local_tool()
        tools.extend(await self._get_mcp_tool())
        tools.extend(await self._get_composio_tools())
        tools.extend(await self._get_langchain_tools())
        return tools

    async def all_tools(self) -> list[dict]:
        return await self._all_tools_async()

    def all_tools_sync(self) -> list[dict]:
        tools: list[dict] = self.get_local_tool()
        if self._client:
            result = asyncio.run(self._get_mcp_tool())
            if result:
                tools.extend(result)
        comp = asyncio.run(self._get_composio_tools())
        if comp:
            tools.extend(comp)
        lc = asyncio.run(self._get_langchain_tools())
        if lc:
            tools.extend(lc)
        return tools

    async def invoke(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_manager: CallbackManager = Inject[CallbackManager],
    ) -> t.Any:
        logger.info("Executing tool '%s' with %d arguments", name, len(args))
        logger.debug("Tool arguments: %s", args)

        event = EventModel.default(
            config,
            data={"args": args, "tool_call_id": tool_call_id, "function_name": name},
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.node_name = name
        publish_event(event)

        if name in self.mcp_tools:
            event.metadata["is_mcp"] = True
            publish_event(event)
            res = await self._mcp_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = res.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            publish_event(event)
            return res

        if name in self.composio_tools:
            event.metadata["is_composio"] = True
            publish_event(event)
            res = await self._composio_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = res.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            publish_event(event)
            return res

        if name in self.langchain_tools:
            event.metadata["is_langchain"] = True
            publish_event(event)
            res = await self._langchain_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = res.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            publish_event(event)
            return res

        if name in self._funcs:
            event.metadata["is_mcp"] = False
            publish_event(event)
            res = await self._internal_execute(
                name,
                args,
                tool_call_id,
                config,
                state,
                callback_manager,
            )
            event.data["message"] = res.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            publish_event(event)
            return res

        error_msg = f"Tool '{name}' not found."
        event.data["error"] = error_msg
        event.event_type = EventType.ERROR
        event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
        publish_event(event)
        return Message.tool_message(
            content=error_msg,
            tool_call_id=tool_call_id,
            is_error=True,
        )

    async def stream(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_manager: CallbackManager = Inject[CallbackManager],
    ) -> t.AsyncIterator[EventModel | Message]:
        logger.info("Executing tool '%s' with %d arguments", name, len(args))
        logger.debug("Tool arguments: %s", args)
        event = EventModel.default(
            config,
            data={"args": args, "tool_call_id": tool_call_id, "function_name": name},
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.node_name = "ToolNode"
        publish_event(event)

        if name in self.mcp_tools:
            event.metadata["is_mcp"] = True
            yield event
            message = await self._mcp_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = message.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            yield event
            yield message
            return

        if name in self.composio_tools:
            event.metadata["is_composio"] = True
            yield event
            message = await self._composio_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = message.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            yield event
            yield message
            return

        if name in self.langchain_tools:
            event.metadata["is_langchain"] = True
            yield event
            message = await self._langchain_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            event.data["message"] = message.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            yield event
            yield message
            return

        if name in self._funcs:
            event.metadata["is_mcp"] = False
            yield event

            result = await self._internal_execute(
                name,
                args,
                tool_call_id,
                config,
                state,
                callback_manager,
            )
            event.data["message"] = result.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            yield event
            yield result
            return

        error_msg = f"Tool '{name}' not found."
        event.data["error"] = error_msg
        event.event_type = EventType.ERROR
        event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
        yield event
        publish_event(event)

        yield Message.tool_message(
            content=error_msg,
            tool_call_id=tool_call_id,
            is_error=True,
        )

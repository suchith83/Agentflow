"""Executors for different tool providers and local functions."""

from __future__ import annotations

import inspect
import json
import logging
import typing as t

from agentflow.core.state import (
    AgentState,
    ContentBlock,
    ErrorBlock,
    Message,
    ToolResult,
    ToolResultBlock,
)
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.utils import CallbackContext, CallbackManager, InvocationType, call_sync_or_async

from .constants import INJECTABLE_PARAMS, has_injected_default


# Sentinel-import: StreamEmitter is imported lazily to avoid circular deps at
# module load time; the type hint is only used in method signatures.
if t.TYPE_CHECKING:
    from agentflow.core.state.stream_emitter import StreamEmitter


logger = logging.getLogger("agentflow.graph.tool_node")

_STATUS_OK: set[str] = {"completed", "success", "ok", "done", "true", "1"}
_STATUS_FAIL: set[str] = {"failed", "failure", "error", "false", "0"}
_ERROR_TRUE: set[str] = {"true", "1", "yes", "error", "failed", "failure"}


def _safe_serialize(obj: t.Any) -> dict[str, t.Any]:
    try:
        json.dumps(obj)
        return obj if isinstance(obj, dict) else {"content": obj}
    except (TypeError, OverflowError):
        if hasattr(obj, "model_dump"):
            dumped = obj.model_dump()  # type: ignore
            if isinstance(dumped, dict) and dumped.get("type") == "resource":
                resource = dumped.get("resource", {})
                if isinstance(resource, dict) and "uri" in resource:
                    resource["uri"] = str(resource["uri"])
                    dumped["resource"] = resource
            return dumped
        return {"content": str(obj), "type": "fallback"}


def _as_bool(val: t.Any, truthy_set: set[str]) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in truthy_set


def _extract_block_meta(
    data: dict[str, t.Any],
) -> tuple[bool, dict[str, t.Any]]:
    """Normalize arbitrary status/error keys; return (is_error, cleaned_data)."""
    data = dict(data)

    raw_status = data.pop("status", None)
    raw_is_error = data.pop("is_error", data.pop("error", None))
    raw_success = data.pop("success", None)

    if raw_is_error is not None:
        is_error = _as_bool(raw_is_error, _ERROR_TRUE)
    elif raw_success is not None:
        is_error = not _as_bool(raw_success, _STATUS_OK)
    else:
        is_error = False

    if raw_status is not None:
        s = str(raw_status).lower()
        if s in _STATUS_FAIL:
            is_error = True

    return is_error, data


class LocalExecMixin:
    _funcs: dict[str, t.Callable]

    def _prepare_input_data_tool(
        self,
        fn: t.Callable,
        name: str,
        args: dict,
        default_data: dict,
    ) -> dict:
        sig = inspect.signature(fn)
        input_data = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param_name in default_data:
                input_data[param_name] = default_data[param_name]
                continue

            if param_name in INJECTABLE_PARAMS:
                continue

            if has_injected_default(param):
                logger.debug(
                    "Skipping injectable parameter '%s' with Inject syntax",
                    param_name,
                )
                continue

            if param_name in args:
                input_data[param_name] = args[param_name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required parameter '{param_name}' for function '{name}'")

        return input_data

    def _publish_internal_completion(
        self,
        event: EventModel,
        message: Message,
        status: str,
    ) -> None:
        event.event_type = EventType.END
        event.data["message"] = message.model_dump()
        event.metadata["status"] = status
        event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
        publish_event(event)

    def _build_internal_result_blocks(
        self,
        result: t.Any,
        tool_call_id: str,
    ) -> list[ContentBlock]:
        is_error = False

        if isinstance(result, str):
            output: str | list[dict[str, t.Any]] = result
        elif isinstance(result, dict):
            is_error, cleaned = _extract_block_meta(result)
            output = [_safe_serialize(cleaned)]
        elif hasattr(result, "model_dump"):
            dumped = result.model_dump()  # type: ignore
            if isinstance(dumped, dict):
                is_error, cleaned = _extract_block_meta(dumped)
                output = [_safe_serialize(cleaned)]
            else:
                output = [_safe_serialize(dumped)]
        elif hasattr(result, "__dict__"):
            is_error, cleaned = _extract_block_meta(result.__dict__)
            output = [_safe_serialize(cleaned)]
        elif isinstance(result, list):
            output = [_safe_serialize(item) for item in result]
        else:
            output = str(result)

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                output=output,
                status="failed" if is_error else "completed",
                is_error=is_error,
            )
        ]

    def _message_from_internal_result(
        self,
        result: t.Any,
        state: AgentState,
        tool_call_id: str,
        meta: dict[str, t.Any],
    ) -> dict[str, t.Any] | Message:
        if isinstance(result, ToolResult):
            if result.state:
                for key, value in result.state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

            return {
                "state": state,
                "messages": Message.tool_message(
                    content=[
                        ToolResultBlock(
                            call_id=tool_call_id,
                            output=result.message,
                            status="failed" if result.is_error else "completed",
                            is_error=result.is_error,
                        )
                    ],
                    meta=meta,
                ),
            }

        if isinstance(result, Message):
            result.metadata = {**meta, **(result.metadata or {})}
            return result

        return Message.tool_message(
            content=self._build_internal_result_blocks(result, tool_call_id),
            meta=meta,
        )

    async def _internal_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        emit: StreamEmitter | None = None,
    ) -> dict[str, t.Any] | Message:
        context = CallbackContext(
            invocation_type=InvocationType.TOOL,
            node_name="ToolNode",
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": config,
            },
        )

        fn = self._funcs[name]
        input_data = self._prepare_input_data_tool(
            fn,
            name,
            args,
            {
                "tool_call_id": tool_call_id,
                "state": state,
                "config": config,
                "emit": emit,
            },
        )

        meta = {
            "function_name": name,
            "function_argument": args,
            "tool_call_id": tool_call_id,
        }

        event = EventModel.default(
            base_config=config,
            data={
                "tool_call_id": tool_call_id,
                "args": args,
                "function_name": name,
                "is_mcp": False,
            },
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.event_type = EventType.PROGRESS
        event.node_name = "ToolNode"
        publish_event(event)

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.UPDATE
            event.metadata["status"] = "before_invoke_complete Invoke internal"
            publish_event(event)

            result = await call_sync_or_async(fn, **input_data)

            result = await callback_mgr.execute_after_invoke(
                context,
                input_data,
                result,
            )

            msg = self._message_from_internal_result(result, state, tool_call_id, meta)
            if isinstance(msg, Message):
                self._publish_internal_completion(event, msg, "Internal tool execution complete")
            elif isinstance(msg, dict):
                self._publish_internal_completion(
                    event,
                    Message.tool_message(
                        content=self._build_internal_result_blocks(
                            msg.get("messages"), tool_call_id
                        ),
                        meta=meta,
                    ),
                    "Internal tool execution complete",
                )
            return msg

        except Exception as e:  # pragma: no cover - error path
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "Internal tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "Internal tool execution complete, with error"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=f"Internal execution error: {e}",
                        status="failed",
                        is_error=True,
                    ),
                    ErrorBlock(message=f"Internal execution error: {e}"),
                ],
                meta=meta,
            )


class MCPMixin:
    _client: t.Any | None
    # The concrete ToolNode defines these
    mcp_tools: list[str]  # type: ignore[assignment]
    _pass_user_info_to_mcp: bool  # type: ignore[assignment]

    def _serialize_result(
        self,
        tool_call_id: str,
        res: t.Any,
    ) -> list[ContentBlock]:
        def safe_serialize(obj: t.Any) -> dict[str, t.Any]:
            try:
                json.dumps(obj)
                return obj if isinstance(obj, dict) else {"content": obj}
            except (TypeError, OverflowError):
                if hasattr(obj, "model_dump"):
                    dumped = obj.model_dump()  # type: ignore
                    if isinstance(dumped, dict) and dumped.get("type") == "resource":
                        resource = dumped.get("resource", {})
                        if isinstance(resource, dict) and "uri" in resource:
                            resource["uri"] = str(resource["uri"])
                            dumped["resource"] = resource
                    return dumped
                return {"content": str(obj), "type": "fallback"}

        for source in [
            getattr(res, "content", None),
            getattr(res, "structured_content", None),
            getattr(res, "data", None),
        ]:
            if source is None:
                continue
            try:
                if isinstance(source, list):
                    result = [safe_serialize(item) for item in source]
                else:
                    result = [safe_serialize(source)]

                return [
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=result,
                        is_error=False,
                        status="completed",
                    )
                ]
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Serialization failure: %s", e)
                continue

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                output=[
                    {
                        "content": str(res),
                        "type": "fallback",
                    }
                ],
                is_error=False,
                status="completed",
            )
        ]

    async def _get_mcp_tool(self, tags: set[str] | None = None) -> list[dict]:
        """Fetch tools from the MCP client, optionally filtering by tags."""
        tools: list[dict] = []
        if not self._client:
            return tools

        try:
            async with self._client:
                res = await self._client.ping()
                if not res:
                    return tools
                mcp_tools: list[t.Any] = await self._client.list_tools()
                for i in mcp_tools:
                    if tags:
                        # Logic: if tags are provided, filter tools by tags
                        meta = i.meta or {}
                        tool_tags = set(meta.get("_fastmcp", {}).get("tags", []))
                        if not tool_tags.intersection(tags):
                            continue
                    # attribute provided by concrete ToolNode
                    self.mcp_tools.append(i.name)  # type: ignore[attr-defined]
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": i.name,
                                "description": i.description,
                                "parameters": i.inputSchema,
                            },
                        }
                    )
        except Exception as e:  # pragma: no cover - network/optional
            logger.exception("Failed to fetch MCP tools: %s", e)

        return tools

    async def _mcp_execute(  # noqa: PLR0915
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        callback_mgr: CallbackManager,
    ) -> Message:
        context = CallbackContext(
            invocation_type=InvocationType.MCP,
            node_name="ToolNode",
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": config,
                "mcp_client": bool(self._client),
            },
        )

        meta = {
            "function_name": name,
            "function_argument": args,
            "tool_call_id": tool_call_id,
        }

        event = EventModel.default(
            base_config=config,
            data={
                "tool_call_id": tool_call_id,
                "args": args,
                "function_name": name,
                "is_mcp": True,
            },
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.event_type = EventType.PROGRESS
        event.node_name = "ToolNode"
        publish_event(event)

        input_data = {**args}

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            event.event_type = EventType.UPDATE
            event.metadata["status"] = "before_invoke_complete Invoke MCP"
            publish_event(event)

            if not self._client:
                error_result = Message.tool_message(
                    content=[
                        ErrorBlock(
                            message="No MCP client configured",
                        ),
                        ToolResultBlock(
                            call_id=tool_call_id,
                            output="No MCP client configured",
                            is_error=True,
                            status="failed",
                        ),
                    ],
                    meta=meta,
                )
                res = await callback_mgr.execute_after_invoke(context, input_data, error_result)
                event.event_type = EventType.ERROR
                event.metadata["error"] = "No MCP client configured"
                publish_event(event)
                return res

            async with self._client:
                if not await self._client.ping():
                    error_result = Message.tool_message(
                        content=[
                            ErrorBlock(message="MCP Server not available. Ping failed."),
                            ToolResultBlock(
                                call_id=tool_call_id,
                                output="MCP Server not available. Ping failed.",
                                is_error=True,
                                status="failed",
                            ),
                        ],
                        meta=meta,
                    )
                    event.event_type = EventType.ERROR
                    event.metadata["error"] = "MCP server not available, ping failed"
                    publish_event(event)
                    return await callback_mgr.execute_after_invoke(
                        context, input_data, error_result
                    )

                if self._pass_user_info_to_mcp:
                    mcp_user_info = config.get("user")
                    if mcp_user_info and isinstance(mcp_user_info, dict):
                        input_data["user"] = mcp_user_info
                    else:
                        user_id = config.get("user_id")
                        if user_id:
                            input_data["user"] = {"user_id": user_id}

                res: t.Any = await self._client.call_tool(name, input_data)

                final_res = self._serialize_result(tool_call_id, res)

                result = Message.tool_message(
                    content=final_res,
                    meta=meta,
                )

                res = await callback_mgr.execute_after_invoke(context, input_data, result)
                event.event_type = EventType.END
                event.data["message"] = result.model_dump()
                event.metadata["status"] = "MCP tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return res

        except Exception as e:  # pragma: no cover - error path
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "MCP tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "MCP tool execution complete, with recovery"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=f"MCP execution error: {e}",
                        is_error=True,
                        status="failed",
                    ),
                    ErrorBlock(message=f"MCP execution error: {e}"),
                ],
                meta=meta,
            )


class KwargsResolverMixin:
    def _should_skip_parameter(self, param: inspect.Parameter) -> bool:
        return param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

    def _handle_injectable_parameter(
        self,
        p_name: str,
        param: inspect.Parameter,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        if p_name in injectable_params:
            injectable_value = injectable_params[p_name]
            if injectable_value is not None:
                return injectable_value

        if dependency_container and dependency_container.has(p_name):
            return dependency_container.get(p_name)

        if param.default is inspect._empty:
            raise TypeError(f"Required injectable parameter '{p_name}' not found")

        return None

    def _get_parameter_value(
        self,
        p_name: str,
        param: inspect.Parameter,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        if p_name in injectable_params:
            return self._handle_injectable_parameter(
                p_name, param, injectable_params, dependency_container
            )

        value_sources = [
            lambda: args.get(p_name),
            lambda: (
                dependency_container.get(p_name)
                if dependency_container and dependency_container.has(p_name)
                else None
            ),
        ]

        for source in value_sources:
            value = source()
            if value is not None:
                return value

        if param.default is not inspect._empty:
            return None

        raise TypeError(f"Missing required parameter '{p_name}' for function")

    def _prepare_kwargs(
        self,
        sig: inspect.Signature,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> dict:
        kwargs: dict = {}
        for p_name, p in sig.parameters.items():
            if self._should_skip_parameter(p):
                continue
            value = self._get_parameter_value(
                p_name, p, args, injectable_params, dependency_container
            )
            if value is not None:
                kwargs[p_name] = value
        return kwargs

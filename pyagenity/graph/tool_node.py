"""Tool node utilities.

Provides a ToolNode that inspects callables and provides JSON-schema-like
descriptions suitable for function-calling LLMs and a simple execute API.
"""

from __future__ import annotations

import inspect
import json
import logging
import typing as t

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp import Tool
from mcp.types import ContentBlock

from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    call_sync_or_async,
    default_callback_manager,
    get_injectable_param_name,
    is_injectable_type,
)
from pyagenity.utils.message import Message


if t.TYPE_CHECKING:
    from pyagenity.checkpointer import BaseCheckpointer
    from pyagenity.publisher import BasePublisher
    from pyagenity.state import AgentState
    from pyagenity.store import BaseStore

logger = logging.getLogger(__name__)


class ToolNode:
    """Registry for callables that exports function specs and executes them."""

    def __init__(
        self,
        functions: t.Iterable[t.Callable],
        client: Client | None = None,
        publisher: BasePublisher | None = None,
    ):
        logger.info("Initializing ToolNode with %d functions", len(list(functions)))
        if client:
            logger.debug("ToolNode initialized with MCP client")
        self._funcs: dict[str, t.Callable] = {}
        self._client: Client | None = client
        self._publisher: BasePublisher | None = publisher
        for fn in functions:
            if not callable(fn):
                error_msg = "ToolNode only accepts callables"
                logger.error(error_msg)
                raise TypeError(error_msg)
            self._funcs[fn.__name__] = fn
            logger.debug("Registered function '%s' in ToolNode", fn.__name__)

        self.mcp_tools = []
        logger.debug("ToolNode initialized with %d local functions", len(self._funcs))

    async def all_tools(self) -> list[dict]:
        """Return function descriptions for all registered callables."""

        tools: list[dict] = []
        logger.debug("Collecting tool descriptions")
        for name, fn in self._funcs.items():
            sig = inspect.signature(fn)
            params_schema: dict = {"type": "object", "properties": {}, "required": []}

            for p_name, p in sig.parameters.items():
                # skip *args/**kwargs
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                # Skip injectable parameters - they shouldn't be in the LLM tool spec
                annotation = p.annotation if p.annotation is not inspect._empty else str
                if is_injectable_type(annotation):
                    continue

                prop = self._annotation_to_schema(annotation, p.default)
                params_schema["properties"][p_name] = prop

                if p.default is inspect._empty:
                    params_schema["required"].append(p_name)

            if not params_schema["required"]:
                params_schema.pop("required")

            description = inspect.getdoc(fn) or "No description provided."

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params_schema,
                    },
                }
            )

        logger.debug("Collected %d local tool descriptions", len(tools))

        # get the tools from client
        logger.debug("Collecting MCP tool descriptions")
        if self._client:
            logger.debug("MCP client is set, fetching tools from MCP server")
            async with self._client:
                # check ping
                res = await self._client.ping()
                # Ping not working, so no need to
                # do anything, return old one
                if not res:
                    return tools

                mcp_tools: list[Tool] = await self._client.list_tools()
                for i in mcp_tools:
                    # also save the names
                    self.mcp_tools.append(i.name)
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

        logger.debug("Collected %d MCP tool descriptions", len(self.mcp_tools))
        return tools

    async def _internal_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState | None = None,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        dependency_container=None,
        callback_manager: CallbackManager | None = None,
    ):
        """Execute internal tool function with callback hooks."""
        callback_mgr = callback_manager or default_callback_manager
        logger.debug("Executing internal tool '%s' with %d arguments", name, len(args))
        logger.info("Executing internal tool '%s'", name)

        # Create callback context for TOOL invocation
        context = CallbackContext(
            invocation_type=InvocationType.TOOL,
            node_name="ToolNode",
            function_name=name,
            metadata={"tool_call_id": tool_call_id, "args": args, "config": config},
        )

        fn = self._funcs[name]
        sig = inspect.signature(fn)

        # Available injectable parameters
        injectable_params = {
            "tool_call_id": tool_call_id,
            "state": state,
            "checkpointer": checkpointer,
            "store": store,
            "config": config,
        }

        kwargs = self._prepare_kwargs(sig, args, injectable_params, dependency_container)

        # Prepare input data for callbacks
        input_data = {
            "function_name": name,
            "args": args,
            "kwargs": kwargs,
            "tool_call_id": tool_call_id,
        }

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            # Extract potentially modified data
            modified_kwargs = input_data.get("kwargs", kwargs)

            # Execute the actual tool function
            logger.debug("Invoking tool function '%s' with kwargs: %s", name, modified_kwargs)
            result = await call_sync_or_async(fn, **modified_kwargs)
            logger.debug("Tool function '%s' returned: %s", name, result)

            # Execute after_invoke callbacks
            return await callback_mgr.execute_after_invoke(context, input_data, result)

        except Exception as e:
            # Execute error callbacks
            logger.error("Error occurred while executing tool '%s': %s", name, e)
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if recovery_result is not None:
                logger.info("Recovery result obtained for tool '%s': %s", name, recovery_result)
                return recovery_result
            # Re-raise the original error
            logger.error("No recovery result for tool '%s', re-raising error", name)
            raise

    def _serialize_result(self, res: CallToolResult) -> str:
        def try_parse_json(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return val
            return val

        def to_obj(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return {"items": val}
            if isinstance(val, ContentBlock):
                obj = val.model_dump()
                # Try to parse the 'text' field if it looks like JSON
                if "text" in obj:
                    obj["text"] = try_parse_json(obj["text"])
                return obj
            if val is not None:
                return val
            return None

        result = []
        if res.content and isinstance(res.content, list):
            for i in res.content:
                ir = to_obj(i)
                if ir is not None:
                    result.append(ir)

        if not result:
            ir = to_obj(res.structured_content)
            if ir is not None:
                result.append(ir)

        if not result:
            ir = to_obj(res.data)
            if ir is not None:
                result.append(ir)

        return json.dumps(result)

    async def _mcp_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState | None = None,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        dependency_container=None,
        callback_manager: CallbackManager | None = None,
    ) -> Message:
        """
        Execute the MCP tool registered under `name` with `args` kwargs.
        Returns a Message with the result or error.
        """
        callback_mgr = callback_manager or default_callback_manager

        # Create callback context for MCP invocation
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

        meta = {"function_name": name, "function_argument": args}
        logger.debug("Executing MCP tool '%s' with %d arguments", name, len(args))
        logger.info("Executing MCP tool '%s'", name)

        # Prepare input data for callbacks
        input_data = {
            "function_name": name,
            "args": args,
            "tool_call_id": tool_call_id,
            "meta": meta,
        }

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            # Extract potentially modified data
            modified_name = input_data.get("function_name", name)
            modified_args = input_data.get("args", args)
            modified_meta = input_data.get("meta", meta)

            if not self._client:
                logger.error("MCP client not set for MCP tool execution")
                error_result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content="MCP Client not Setup",
                    is_error=True,
                    meta=modified_meta,
                )
                # Execute after_invoke callbacks even for errors
                return await callback_mgr.execute_after_invoke(context, input_data, error_result)

            async with self._client:
                logger.debug("Pinging MCP server")
                if not await self._client.ping():
                    logger.error("MCP server not available. Ping failed.")
                    error_result = Message.tool_message(
                        tool_call_id=tool_call_id,
                        content="MCP Server not available. Ping failed.",
                        is_error=True,
                        meta=modified_meta,
                    )
                    # Execute after_invoke callbacks even for errors
                    return await callback_mgr.execute_after_invoke(
                        context, input_data, error_result
                    )

                logger.debug("Calling MCP tool '%s' with args: %s", modified_name, modified_args)
                res: CallToolResult = await self._client.call_tool(modified_name, modified_args)
                logger.debug("MCP tool '%s' returned: %s", modified_name, res)
                final_res = self._serialize_result(res)

                result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=final_res,
                    is_error=bool(res.is_error),
                    meta=modified_meta,
                )

                # Execute after_invoke callbacks
                return await callback_mgr.execute_after_invoke(context, input_data, result)

        except Exception as e:
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)
            logger.error("Error occurred while executing MCP tool '%s': %s", name, e)

            if recovery_result is not None:
                logger.info("Recovery result obtained for tool '%s': %s", name, recovery_result)
                return recovery_result

            # Return error message if no recovery
            logger.error("No recovery result for tool '%s', re-raising error", name)
            return Message.tool_message(
                tool_call_id=tool_call_id,
                content=f"MCP execution error: {e}",
                is_error=True,
                meta=meta,
            )

    async def execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState | None = None,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        dependency_container=None,
        callback_manager: CallbackManager | None = None,
    ) -> t.Any:
        """Execute the callable registered under `name` with `args` kwargs.

        Additional injectable parameters:
        - tool_call_id: ID of the tool call (can be injected into function if needed)
        - state: Current agent state (can be injected into function if needed)
        - checkpointer: Checkpointer instance (can be injected into function if needed)
        - store: Store instance (can be injected into function if needed)
        - dependency_container: Container with custom dependencies
        - callback_manager: Callback manager for executing hooks
        """
        logger.info("Executing tool '%s' with %d arguments", name, len(args))
        logger.debug("Tool arguments: %s", args)

        # check in mcp
        if name in self.mcp_tools:
            logger.debug("Tool '%s' found in MCP tools, executing via MCP", name)
            return await self._mcp_execute(
                name,
                args,
                tool_call_id,
                config,
                state,
                checkpointer,
                store,
                dependency_container,
                callback_manager,
            )

        if name in self._funcs:
            logger.debug("Tool '%s' found in local functions, executing internally", name)
            return await self._internal_execute(
                name,
                args,
                tool_call_id,
                config,
                state,
                checkpointer,
                store,
                dependency_container,
                callback_manager,
            )

        error_msg = f"Tool '{name}' not found."
        logger.warning(error_msg)
        return Message.tool_message(
            content=error_msg,
            tool_call_id=tool_call_id,
            is_error=True,
        )

    def _prepare_kwargs(
        self,
        sig: inspect.Signature,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> dict:
        """Prepare keyword arguments for function execution."""
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

    def _should_skip_parameter(self, param: inspect.Parameter) -> bool:
        """Check if parameter should be skipped (VAR_POSITIONAL or VAR_KEYWORD)."""
        return param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

    def _get_parameter_value(
        self,
        p_name: str,
        param: inspect.Parameter,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        """Get the value for a parameter from various sources."""
        # Check if this parameter should be injected based on type annotation
        annotation = param.annotation if param.annotation is not inspect._empty else None

        if annotation and is_injectable_type(annotation):
            return self._handle_injectable_parameter(
                p_name, param, annotation, injectable_params, dependency_container
            )

        # Try different value sources in order of priority
        value_sources = [
            lambda: args.get(p_name),  # Function arguments
            lambda: (
                injectable_params.get(p_name) if injectable_params.get(p_name) is not None else None
            ),  # Injectable params
            lambda: (
                dependency_container.get(p_name)
                if dependency_container and dependency_container.has(p_name)
                else None
            ),  # Dependency container
        ]

        for source in value_sources:
            value = source()
            if value is not None:
                return value

        # Handle default or raise error
        if param.default is not inspect._empty:
            return None  # Use default

        raise TypeError(f"Missing required parameter '{p_name}' for function")

    def _handle_injectable_parameter(
        self,
        p_name: str,
        param: inspect.Parameter,
        annotation: t.Any,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        """Handle parameter injection based on type annotation."""
        injectable_param_name = get_injectable_param_name(annotation)

        if injectable_param_name == "dependency":
            if dependency_container and dependency_container.has(p_name):
                return dependency_container.get(p_name)
            if param.default is inspect._empty:
                raise TypeError(f"Required dependency '{p_name}' not found in container")
            return None  # Use default

        if injectable_param_name and injectable_param_name in injectable_params:
            injectable_value = injectable_params[injectable_param_name]
            if injectable_value is not None:
                return injectable_value

        return None

    @staticmethod
    def _annotation_to_schema(annotation: t.Any, default: t.Any) -> dict:
        """Map simple Python annotations to JSON-schema-like dicts.

        Supports basic primitives, list[...] and typing.Literal for enums.
        Falls back to string when unknown.
        """
        # Handle Optional[...] / Union[..., None]
        schema = ToolNode._handle_optional_annotation(annotation, default)
        if schema:
            return schema

        # Map primitive types
        primitive_mappings = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }

        if annotation in primitive_mappings:
            schema = primitive_mappings[annotation]
        else:
            schema = ToolNode._handle_complex_annotation(annotation)

        # Add default if present
        if default is not inspect._empty:
            schema["default"] = default

        return schema

    @staticmethod
    def _handle_optional_annotation(annotation: t.Any, default: t.Any) -> dict | None:
        """Handle Optional[...] / Union[..., None] annotations."""
        args = getattr(annotation, "__args__", None)
        if args and any(a is type(None) for a in args):
            # pick the non-None arg and map that
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return ToolNode._annotation_to_schema(non_none[0], default)
        return None

    @staticmethod
    def _handle_complex_annotation(annotation: t.Any) -> dict:
        """Handle complex annotations like list[...] and Literal[...]."""
        origin = getattr(annotation, "__origin__", None)

        # Handle list types
        if origin is list:
            item_type = getattr(annotation, "__args__", (str,))[0]
            item_schema = ToolNode._annotation_to_schema(item_type, None)
            return {"type": "array", "items": item_schema}

        # Handle Literal types
        Literal = getattr(t, "Literal", None)
        if Literal is not None and origin is Literal:
            literals = list(getattr(annotation, "__args__", ()))
            if all(isinstance(literal, str) for literal in literals):
                return {"type": "string", "enum": literals}
            return {"enum": literals}

        # Default fallback
        return {"type": "string"}

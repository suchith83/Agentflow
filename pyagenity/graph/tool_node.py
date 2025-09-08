"""Tool node utilities.

Provides a ToolNode that inspects callables and provides JSON-schema-like
descriptions suitable for function-calling LLMs and a simple execute API.
"""

from __future__ import annotations

import inspect
import json
import logging
import typing as t

from pyagenity.utils.streamming import StreamChunk, StreamEvent


try:
    from fastmcp import Client
    from fastmcp.client.client import CallToolResult

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    Client = None  # type: ignore
    CallToolResult = None  # type: ignore

try:
    from mcp import Tool
    from mcp.types import ContentBlock

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Tool = None  # type: ignore
    ContentBlock = None  # type: ignore

from injectq import inject

from pyagenity.state import AgentState
from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    call_sync_or_async,
)
from pyagenity.utils.message import Message


logger = logging.getLogger(__name__)


INJECTABLE_PARAMS = {
    "tool_call_id",
    "state",
    "config",
    "generated_id",
    "context_manager",
    "publisher",
    "checkpointer",
    "store",
}


class ToolNode:
    """Registry for callables that exports function specs and executes them.

    MCP support requires: pip install pyagenity[fastapi]
    """

    def __init__(
        self,
        functions: t.Iterable[t.Callable],
        client: t.Any | None = None,
    ):
        logger.info("Initializing ToolNode with %d functions", len(list(functions)))

        # Check MCP dependencies if client is provided
        if client is not None:
            if not HAS_FASTMCP or not HAS_MCP:
                raise ImportError(
                    "MCP client functionality requires 'fastmcp' and 'mcp' packages. "
                    "Install with: pip install pyagenity[mcp]"
                )
            logger.debug("ToolNode initialized with MCP client")

        self._funcs: dict[str, t.Callable] = {}
        self._client: t.Any | None = client
        for fn in functions:
            if not callable(fn):
                error_msg = "ToolNode only accepts callables"
                logger.error(error_msg)
                raise TypeError(error_msg)
            self._funcs[fn.__name__] = fn
            logger.debug("Registered function '%s' in ToolNode", fn.__name__)

        self.mcp_tools = []
        logger.debug("ToolNode initialized with %d local functions", len(self._funcs))

    async def _get_local_tool(self) -> list[dict]:
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
                if p_name in INJECTABLE_PARAMS:
                    continue

                annotation = p.annotation if p.annotation is not inspect._empty else str
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
        return tools

    async def _get_mcp_tool(self) -> list[dict]:
        tools: list[dict] = []
        logger.debug("Collecting MCP tool descriptions")
        if self._client:
            logger.debug("MCP client is set, fetching tools from MCP server")
            async with self._client:
                # check ping
                res = await self._client.ping()
                # Ping not working, so no need to
                # do anything, return old one
                if not res:
                    logger.error("MCP server not available. Ping failed.")
                    return tools

                mcp_tools: list[t.Any] = await self._client.list_tools()
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

    async def all_tools(self) -> list[dict]:
        """Return function descriptions for all registered callables."""
        tools: list[dict] = await self._get_local_tool()
        tools.extend(await self._get_mcp_tool())
        return tools

    def _prepare_input_data_tool(
        self,
        fn: t.Callable,
        name: str,
        args: dict,
        default_data: dict,
    ) -> dict:
        sig = inspect.signature(fn)
        input_data = {}
        # # Get injectable parameters to determine which ones to exclude from manual passing
        # # Prepare function arguments (excluding injectable parameters)
        for param_name, param in sig.parameters.items():
            # Skip *args/**kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # check its state, config, tool_call_id
            if param_name in ["state", "config", "tool_call_id"]:
                input_data[param_name] = default_data[param_name]
                continue

            # Skip injectable parameters - they will be handled by injectq
            if param_name in INJECTABLE_PARAMS:
                continue

            # Include regular function arguments
            if param_name in args:
                input_data[param_name] = args[param_name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required parameter '{param_name}' for function '{name}'")

        return input_data

    async def _internal_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_mgr: CallbackManager,
    ) -> Message:
        """Execute internal tool function with callback hooks."""
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
        input_data = self._prepare_input_data_tool(
            fn,
            name,
            args,
            {
                "tool_call_id": tool_call_id,
                "state": state,
                "config": config,
            },
        )

        try:
            # Execute before_invoke callbacks
            meta = {"function_name": name, "function_argument": args}
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            # Execute the actual tool function with injectq handling dependency injection
            logger.debug("Invoking tool function '%s'", name)
            result = await call_sync_or_async(fn, **input_data)
            logger.debug("Tool function '%s' returned: %s", name, result)

            # Prepare tool result message to the callback
            result = await callback_mgr.execute_after_invoke(
                context,
                input_data,
                result,
            )

            # Handle different return types
            if isinstance(result, Message):
                logger.debug("Node '%s' tool execution returned a Message", name)
                # lets update the meta
                meta_data = result.metadata or {}
                meta.update(meta_data)
                result.metadata = meta
                return result

            if isinstance(result, str):
                logger.debug("Node '%s' tool execution returned a string", name)
                # Convert string result to tool message with tool_call_id
                return Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=result,
                    meta=meta,
                )

            # Convert other types to string then to tool message
            logger.debug(
                "Node '%s' tool execution returned an unexpected type: %s",
                name,
                type(result),
            )
            serialized_result = result
            if isinstance(result, dict):
                serialized_result = json.dumps(result)
            elif hasattr(result, "model_dump"):
                serialized_result = json.dumps(result.model_dump())
            elif hasattr(result, "__dict__"):
                serialized_result = json.dumps(result.__dict__)
            elif isinstance(result, list):
                serialized_result = json.dumps(result)
            elif not isinstance(result, str):
                serialized_result = str(result)

            return Message.tool_message(
                tool_call_id=tool_call_id,
                content=serialized_result,
                meta=meta,
            )

        except Exception as e:
            # Execute error callbacks
            logger.error("Error occurred while executing tool '%s': %s", name, e)
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                logger.info("Recovery result obtained for tool '%s': %s", name, recovery_result)
                return recovery_result

            # Re-raise the original error
            logger.error(
                "No recovery result for tool '%s', re-raising error, Please return Message",
                name,
            )
            raise

    def _serialize_result(self, res: t.Any) -> str:
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
            if isinstance(val, ContentBlock):  # type: ignore
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
        callback_mgr: CallbackManager,
    ) -> Message:
        """
        Execute the MCP tool registered under `name` with `args` kwargs.
        Returns a Message with the result or error.
        """

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

        meta = {"function_name": name, "function_argument": args, "tool_call_id": tool_call_id}
        logger.debug("Executing MCP tool '%s' with %d arguments", name, len(args))
        logger.info("Executing MCP tool '%s'", name)

        # Prepare input data for callbacks
        input_data = {**args}

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            if not self._client:
                logger.error("MCP client not set for MCP tool execution")
                error_result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content="MCP Client not Setup",
                    is_error=True,
                    meta=meta,
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
                        meta=meta,
                    )
                    # Execute after_invoke callbacks even for errors
                    return await callback_mgr.execute_after_invoke(
                        context, input_data, error_result
                    )

                logger.debug("Calling MCP tool '%s'", name)
                ############################################
                ############ Call the MCP tool #############
                ############################################
                res: t.Any = await self._client.call_tool(name, input_data)
                logger.debug("MCP tool '%s' returned: %s", name, res)
                ############################################
                ############ Call the MCP tool #############
                ############################################

                final_res = self._serialize_result(res)

                result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=final_res,
                    is_error=bool(res.is_error),
                    meta=meta,
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
        # Check if this parameter should be injected based on parameter name
        if p_name in injectable_params:
            return self._handle_injectable_parameter(
                p_name, param, injectable_params, dependency_container
            )

        # Try different value sources in order of priority
        value_sources = [
            lambda: args.get(p_name),  # Function arguments
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
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        """Handle parameter injection based on parameter name."""
        # Check if it's a known injectable parameter
        if p_name in injectable_params:
            injectable_value = injectable_params[p_name]
            if injectable_value is not None:
                return injectable_value

        # Check if it's a dependency that should be injected from the container
        if dependency_container and dependency_container.has(p_name):
            return dependency_container.get(p_name)

        # If no value found and parameter has no default, raise error
        if param.default is inspect._empty:
            raise TypeError(f"Required injectable parameter '{p_name}' not found")

        return None  # Use default

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

    @inject
    async def invoke(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_manager: CallbackManager,  # type: ignore
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
                callback_manager,
            )

        error_msg = f"Tool '{name}' not found."
        logger.warning(error_msg)
        return Message.tool_message(
            content=error_msg,
            tool_call_id=tool_call_id,
            is_error=True,
        )

    @inject
    async def stream(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_manager: CallbackManager,  # type: ignore
    ) -> t.AsyncIterable[StreamChunk | Message]:
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
        run_id = config.get("run_id", "")
        data = {"args": args, "tool_call_id": tool_call_id, "config": config}
        cfg = {
            "thread_id": config.get("thread_id", ""),
            "run_id": run_id,
            "run_timestamp": config.get("timestamp", ""),
        }

        # check in mcp
        if name in self.mcp_tools:
            logger.debug("Tool '%s' found in MCP tools, executing via MCP", name)
            yield StreamChunk(
                event=StreamEvent.MCP_TOOL_EXECUTION,
                event_type="Before",
                run_id=run_id,
                data=data,
                metadata=cfg,
            )
            message = await self._mcp_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            # pass two events - tool result and message, message will be saved as it is
            # And steam chunk will be used for streaming UIs
            data["message"] = message.model_dump()
            yield StreamChunk(
                event=StreamEvent.MCP_TOOL_RESULT,
                event_type="After",
                run_id=run_id,
                data=data,
                metadata=cfg,
            )
            yield message

        if name in self._funcs:
            logger.debug("Tool '%s' found in local functions, executing internally", name)
            yield StreamChunk(
                event=StreamEvent.TOOL_EXECUTION,
                event_type="Before",
                run_id=run_id,
                data=data,
                metadata=cfg,
            )
            result = await self._internal_execute(
                name,
                args,
                tool_call_id,
                config,
                state,
                callback_manager,
            )

            data["message"] = result.model_dump()
            yield StreamChunk(
                event=StreamEvent.TOOL_EXECUTION,
                event_type="Before",
                run_id=run_id,
                data=data,
                metadata=cfg,
            )

        error_msg = f"Tool '{name}' not found."
        logger.warning(error_msg)
        data["error"] = error_msg
        yield StreamChunk(
            event=StreamEvent.TOOL_RESULT,
            event_type="After",
            run_id=run_id,
            data=data,
            is_error=True,
            metadata=cfg,
        )
        yield Message.tool_message(
            content=error_msg,
            tool_call_id=tool_call_id,
            is_error=True,
        )

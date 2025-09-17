"""Tool node utilities.

Provides a ToolNode that inspects callables and provides JSON-schema-like
descriptions suitable for function-calling LLMs and a simple execute API.
"""

from __future__ import annotations

import inspect
import json
import logging
import typing as t
import asyncio
from pyagenity.graph.utils.utils import publish_event
from pyagenity.utils.streaming import ContentType, Event, EventModel, EventType


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

from injectq import Inject

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
    """Registry for callables that exposes function specs and executes them.

    This class inspects Python callables and provides JSON-schema-like
    descriptions suitable for function-calling LLMs. It also provides
    execution helpers for local functions and MCP-backed tools, including
    callback hooks and event publishing for progress and results.

    MCP support requires the optional dependencies 'fastmcp' and 'mcp'.

    Attributes:
        _funcs: Mapping of function name to callable for locally-registered
            tools.
        _client: Optional MCP client instance used to call remote MCP tools.
        mcp_tools: List of tool names available on the MCP server.
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

    def _get_local_tool(self) -> list[dict]:
        """Build JSON-schema-like descriptions for locally-registered callables.

        The returned list contains entries compatible with function-calling LLM
        formats. Injectable parameters (e.g. 'state', 'publisher') are omitted
        from the schema because they are injected at runtime rather than
        provided by an external caller.

        Returns:
            A list of dictionaries, each with the structure required for a
            function-calling tool description (``type: function`` with a
            nested ``function`` object containing ``name``, ``description`` and
            ``parameters``).
        """
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
        """Fetch tool descriptions from an MCP server using the configured client.

        If an MCP client was provided to the constructor, this method opens a
        short-lived client session and queries the server for available tools.
        Each MCP tool is converted into the same function description format
        used for local tools.

        Returns:
            A list of function description dicts for tools available on the
            MCP server. If the MCP client is not configured or the ping fails,
            an empty list is returned.
        """
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
        """Return function descriptions for all registered callables.

        This aggregates descriptions for both local functions registered with
        this ToolNode and tools discovered on the MCP server (if configured).

        Returns:
            A list of function description dictionaries suitable for passing to
            an LLM that supports function calling.
        """
        tools: list[dict] = self._get_local_tool()
        tools.extend(await self._get_mcp_tool())
        return tools

    def all_tools_sync(self) -> list[dict]:
        """Return function descriptions for all registered callables.

        This aggregates descriptions for both local functions registered with
        this ToolNode and tools discovered on the MCP server (if configured).

        Returns:
            A list of function description dictionaries suitable for passing to
            an LLM that supports function calling.
        """
        tools: list[dict] = self._get_local_tool()
        if not self._client:
            return tools

        result = asyncio.run(self._get_mcp_tool())
        if result:
            tools.extend(result)

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

            # Check if parameter uses Inject[...] syntax for dependency injection
            if (
                hasattr(param, "default")
                and param.default is not inspect._empty
                and hasattr(param.default, "__class__")
            ):
                try:
                    # Check if default value is an Inject instance
                    if "Inject" in str(type(param.default)):
                        logger.debug(
                            "Skipping injectable parameter '%s' with Inject syntax", param_name
                        )
                        continue
                except Exception as e:
                    logger.debug(
                        "Could not determine if parameter '%s' uses Inject: %s", param_name, e
                    )

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
        """Execute a locally-registered tool and publish lifecycle events.

        This method orchestrates execution of a local Python callable that has
        been registered with this ToolNode. It performs the following steps:
        1. Builds the input kwargs for the callable, excluding injectable
           parameters that are provided via ``default_data``.
        2. Publishes a PROGRESS event indicating the tool call has started.
        3. Executes ``before_invoke`` callbacks.
        4. Calls the target function (sync or async) and runs ``after_invoke``
           callbacks on the result.
        5. Publishes UPDATE/END/ERROR events as appropriate and returns a
           ``Message`` representing the result.

        Args:
            name: Registered name of the local tool to invoke.
            args: Arguments provided by the caller.
            tool_call_id: Unique identifier for this tool invocation.
            config: Configuration mapping used when constructing EventModel
                instances.
            state: AgentState instance to inject into the callable if it
                declares a ``state`` parameter.
            callback_mgr: Callback manager used to execute lifecycle
                callbacks.
            publisher: Optional publisher used to emit EventModel updates.

        Returns:
            A ``Message`` representing the tool result. On error, a
            Message with ``is_error=True`` will be returned if no recovery
            callback provides an alternate result.
        """
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

        meta = {"function_name": name, "function_argument": args, "tool_call_id": tool_call_id}

        # Create and publish initial progress event
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
        event.sequence_id = 1
        publish_event(event)

        try:
            # Execute before_invoke callbacks
            meta = {"function_name": name, "function_argument": args}
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.UPDATE
            event.sequence_id = 2
            event.metadata["status"] = "before_invoke_complete Invoke internal"
            publish_event(event)

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
                # update meta and publish end event
                meta_data = result.metadata or {}
                meta.update(meta_data)
                result.metadata = meta

                event.event_type = EventType.END
                event.data["message"] = result.model_dump()
                event.metadata["status"] = "Internal tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return result

            if isinstance(result, str):
                logger.debug("Node '%s' tool execution returned a string", name)
                msg = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content=result,
                    meta=meta,
                )
                event.event_type = EventType.END
                event.data["message"] = msg.model_dump()
                event.metadata["status"] = "Internal tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return msg

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

            msg = Message.tool_message(
                tool_call_id=tool_call_id,
                content=serialized_result,
                meta=meta,
            )

            event.event_type = EventType.END
            event.data["message"] = msg.model_dump()
            event.metadata["status"] = "Internal tool execution complete"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            publish_event(event)

            return msg

        except Exception as e:
            # Execute error callbacks
            logger.error("Error occurred while executing tool '%s': %s", name, e)
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                logger.info("Recovery result obtained for tool '%s': %s", name, recovery_result)
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "Internal tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            # Return error message if no recovery
            logger.error("No recovery result for tool '%s', returning error message", name)
            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "Internal tool execution complete, with error"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                tool_call_id=tool_call_id,
                content=f"Internal execution error: {e}",
                is_error=True,
                meta=meta,
            )

    def _serialize_result(self, res: t.Any) -> str:
        """Serialize an MCP CallToolResult-like object to a JSON string.

        Args:
            res: The result object returned from an MCP client call. The
                object may expose ``content``, ``structured_content`` or
                ``data`` fields. Content blocks may be of type ``ContentBlock``
                which this function will convert via ``model_dump``.

        Returns:
            A JSON string representing a list of parsed result objects.
        """

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
        """Execute a remote MCP tool via the configured MCP client.

        The method publishes lifecycle events (PROGRESS, UPDATE, END, ERROR)
        that describe the execution of the MCP tool. It also runs lifecycle
        callbacks (``before_invoke``, ``after_invoke``, ``execute_on_error``)
        so external code can react to or modify inputs/results/errors.

        Args:
            name: MCP tool name.
            args: Arguments passed to the MCP tool.
            tool_call_id: Unique identifier for this invocation.
            config: Configuration mapping used when creating EventModel
                instances.
            callback_mgr: Callback manager used to execute lifecycle
                callbacks.
            publisher: Optional event publisher used to emit EventModel
                updates.

        Returns:
            A ``Message`` representing the final result or an error.
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
        event.sequence_id = 1
        publish_event(event)

        # Prepare input data for callbacks
        input_data = {**args}

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            event.event_type = EventType.UPDATE
            event.sequence_id = 2
            event.metadata["status"] = "before_invoke_complete Invoke MCP"
            publish_event(event)

            if not self._client:
                logger.error("MCP client not set for MCP tool execution")
                error_result = Message.tool_message(
                    tool_call_id=tool_call_id,
                    content="MCP Client not Setup",
                    is_error=True,
                    meta=meta,
                )
                # Execute after_invoke callbacks even for errors
                res = await callback_mgr.execute_after_invoke(context, input_data, error_result)
                event.event_type = EventType.ERROR
                event.metadata["error"] = "No MCP client configured"
                publish_event(event)
                return res

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
                    event.event_type = EventType.ERROR
                    event.metadata["error"] = "MCP server not available, ping failed"
                    publish_event(event)
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
                res = await callback_mgr.execute_after_invoke(context, input_data, result)
                event.event_type = EventType.END
                event.data["message"] = result.model_dump()
                event.metadata["status"] = "MCP tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return res

        except Exception as e:
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)
            logger.error("Error occurred while executing MCP tool '%s': %s", name, e)

            if isinstance(recovery_result, Message):
                logger.info("Recovery result obtained for tool '%s': %s", name, recovery_result)
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "MCP tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            # Return error message if no recovery
            logger.error("No recovery result for tool '%s', re-raising error", name)
            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "MCP tool execution complete, with recovery"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

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
        """Prepare keyword arguments for a callable from multiple sources.

        This inspects ``sig`` and resolves values for each parameter using the
        following precedence:
        1. Explicit ``args`` provided by the caller
        2. Values available from ``dependency_container`` (if provided)
        3. Injectable parameters from ``injectable_params``

        Args:
            sig: The inspected function signature.
            args: Caller-provided arguments.
            injectable_params: A mapping of injectable parameter names to
                their runtime values.
            dependency_container: Optional container object that exposes
                ``has(name)`` and ``get(name)`` for dependency lookup.

        Returns:
            A dict of keyword arguments ready to be passed to the target
            callable.
        """
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
        """Determine whether a parameter should be skipped when preparing kwargs.

        Skipped parameters include var-positionals (``*args``) and var-keywords
        (``**kwargs``) since they are not represented in the function schema
        and cannot be populated by name.

        Args:
            param: The inspect.Parameter to examine.

        Returns:
            True if the parameter should be skipped, False otherwise.
        """
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
        """Resolve the value for a single parameter from configured sources.

        The resolution order is:
        1. If the parameter is listed in ``injectable_params``, use the
           injectable handling function.
        2. Look up the name in the explicit ``args`` mapping.
        3. Look up the name in the ``dependency_container`` if available.
        4. If a default exists on the parameter, return ``None`` to indicate
           the caller should use the default.
        5. Otherwise raise ``TypeError`` for a missing required parameter.

        Args:
            p_name: Parameter name.
            param: The inspect.Parameter object.
            args: Caller-provided arguments.
            injectable_params: Mapping of injectable names to runtime values.
            dependency_container: Optional dependency container.

        Returns:
            The resolved value, or ``None`` when the parameter has a default
            and the default should be used.

        Raises:
            TypeError: If a required parameter cannot be resolved.
        """
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
        """Provide a value for parameters that are considered injectable.

        This will first return a value explicitly provided in
        ``injectable_params``. If none is present, the method attempts to
        resolve the dependency from ``dependency_container``. If neither
        yields a value and the parameter has no default, a ``TypeError`` is
        raised.

        Args:
            p_name: Name of the parameter to resolve.
            param: The inspect.Parameter instance describing the parameter.
            injectable_params: Mapping of injectable parameter names to
                runtime values.
            dependency_container: Optional dependency container exposing
                ``has`` and ``get`` methods.

        Returns:
            The injectable value or ``None`` to indicate the function should
            use the parameter's default value.

        Raises:
            TypeError: If an injectable parameter is required but missing.
        """
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
        """Handle Optional[...] / Union[..., None] annotations.

        Returns the schema for the non-None member type when the annotation is
        optional, otherwise ``None``.
        """
        args = getattr(annotation, "__args__", None)
        if args and any(a is type(None) for a in args):
            # pick the non-None arg and map that
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return ToolNode._annotation_to_schema(non_none[0], default)
        return None

    @staticmethod
    def _handle_complex_annotation(annotation: t.Any) -> dict:
        """Handle complex annotations like list[...] and Literal[...].

        This resolves list item types and converts ``typing.Literal`` into an
        ``enum`` description where possible.
        """
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

    async def invoke(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_manager: CallbackManager = Inject[CallbackManager],
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

        event = EventModel.default(
            config,
            data={"args": args, "tool_call_id": tool_call_id, "function_name": name},
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.node_name = name
        publish_event(event)

        # check in mcp
        if name in self.mcp_tools:
            logger.debug("Tool '%s' found in MCP tools, executing via MCP", name)
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

        if name in self._funcs:
            logger.debug("Tool '%s' found in local functions, executing internally", name)
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
        logger.warning(error_msg)
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
        event = EventModel.default(
            config,
            data={"args": args, "tool_call_id": tool_call_id, "function_name": name},
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.node_name = "ToolNode"
        publish_event(event)

        # check in mcp
        if name in self.mcp_tools:
            logger.debug("Tool '%s' found in MCP tools, executing via MCP", name)
            event.metadata["is_mcp"] = True
            yield event
            message = await self._mcp_execute(
                name,
                args,
                tool_call_id,
                config,
                callback_manager,
            )
            # pass two events - tool result and message, message will be saved as it is
            # And steam chunk will be used for streaming UIs
            event.data["message"] = message.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            # first yield the event
            yield event
            # then yield the message
            yield message
            return

        if name in self._funcs:
            logger.debug("ENTERING IF BLOCK for tool '%s'", name)
            logger.debug("Tool '%s' found in local functions, executing internally", name)
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
            # Now we are going to yield two events - tool result and message,
            event.data["message"] = result.model_dump()
            event.event_type = EventType.END
            event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
            # first yield the event
            yield event
            # then yield the message
            yield result
            return

        error_msg = f"Tool '{name}' not found."
        logger.warning(error_msg)
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

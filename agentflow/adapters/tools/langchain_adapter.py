"""LangChain adapter for agentflow (generic wrapper, registry-based).

This adapter mirrors the spirit of Google's ADK LangChain wrapper by allowing
you to register any LangChain tool (BaseTool/StructuredTool) or a duck-typed
object that exposes a ``run``/``_run`` method, then exposing it to agentflow in
the uniform function-calling schema that ``ToolNode`` expects.

Key points:
- Register arbitrary tools at runtime via ``register_tool`` / ``register_tools``.
- Tool schemas are derived from ``tool.args`` (when available) or inferred from
    the tool's pydantic ``args_schema``; otherwise, we fallback to a minimal
    best-effort schema inferred from the wrapped function signature.
- Execution prefers ``invoke`` (Runnable interface) and falls back to ``run``/
    ``_run`` or calling a wrapped function with kwargs.

Optional install:
        pip install 10xscale-agentflow[langchain]

Backward-compat convenience:
- For continuity with prior versions, the adapter can auto-register two common
    tools (``tavily_search`` and ``requests_get``) if ``autoload_default_tools`` is
    True and no user-registered tools exist. You can disable this by passing
    ``autoload_default_tools=False`` to the constructor.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import typing as t


logger = logging.getLogger(__name__)


try:
    import importlib.util as _util

    HAS_LANGCHAIN = _util.find_spec("langchain_core") is not None
except Exception:
    HAS_LANGCHAIN = False


class LangChainToolWrapper:
    """
    Wrap a LangChain tool or a duck-typed tool into a uniform interface.

    Responsibilities:
        - Resolve execution entrypoint (invoke/run/_run/callable func)
        - Provide a function-calling schema {name, description, parameters}
        - Execute with dict arguments and return a JSON-serializable result
    """

    def __init__(
        self,
        tool: t.Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Initialize LangChainToolWrapper.

        Args:
            tool (Any): The LangChain tool or duck-typed object to wrap.
            name (str | None): Optional override for tool name.
            description (str | None): Optional override for tool description.
        """
        self._tool = tool
        self.name = name or getattr(tool, "name", None) or self._default_name(tool)
        self.description = (
            description
            or getattr(tool, "description", None)
            or f"LangChain tool wrapper for {type(tool).__name__}"
        )
        self._callable = self._resolve_callable(tool)

    @staticmethod
    def _default_name(tool: t.Any) -> str:
        # Prefer class name in snake_case-ish
        cls = type(tool).__name__
        return cls[0].lower() + "".join((c if c.islower() else f"_{c.lower()}") for c in cls[1:])

    @staticmethod
    def _resolve_callable(tool: t.Any) -> t.Callable[..., t.Any] | None:
        # Try StructuredTool.func or coroutine
        try:
            # Avoid importing StructuredTool; duck-type attributes
            if getattr(tool, "func", None) is not None:
                return t.cast(t.Callable[..., t.Any], tool.func)
            if getattr(tool, "coroutine", None) is not None:
                return t.cast(t.Callable[..., t.Any], tool.coroutine)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Ignoring tool callable resolution error: %s", exc)
        # Fallback to run/_run methods as callables
        if hasattr(tool, "_run"):
            return tool._run  # type: ignore[attr-defined]
        if hasattr(tool, "run"):
            return tool.run  # type: ignore[attr-defined]
        # Nothing callable to directly use; rely on invoke/run on execution
        return None

    def _json_schema_from_args_schema(self) -> dict[str, t.Any] | None:
        # LangChain BaseTool typically provides .args (already JSON schema)
        schema = getattr(self._tool, "args", None)
        if isinstance(schema, dict) and schema.get("type") == "object":
            return schema

        # Try args_schema (pydantic v1 or v2)
        args_schema = getattr(self._tool, "args_schema", None)
        if args_schema is None:
            return None
        try:
            # pydantic v2
            if hasattr(args_schema, "model_json_schema"):
                js = args_schema.model_json_schema()  # type: ignore[attr-defined]
            else:  # pydantic v1
                js = args_schema.schema()  # type: ignore[attr-defined]
            # Convert typical pydantic schema to a plain "type: object" with properties
            # Look for properties directly
            props = js.get("properties") or {}
            required = js.get("required") or []
            return {"type": "object", "properties": props, "required": required}
        except Exception:  # pragma: no cover - be tolerant
            return None

    def _infer_schema_from_signature(self) -> dict[str, t.Any]:
        func = self._callable or getattr(self._tool, "invoke", None)
        if func is None or not callable(func):  # last resort empty schema
            return {"type": "object", "properties": {}}

        try:
            sig = inspect.signature(func)
            properties: dict[str, dict[str, t.Any]] = {}
            required: list[str] = []
            for name, param in sig.parameters.items():
                if name in {"self", "run_manager", "config", "callbacks"}:
                    continue
                ann = param.annotation
                json_type: str | None = None
                if ann is not inspect._empty:  # type: ignore[attr-defined]
                    json_type = self._map_annotation_to_json_type(ann)
                prop: dict[str, t.Any] = {}
                if json_type:
                    prop["type"] = json_type
                if param.default is inspect._empty:  # type: ignore[attr-defined]
                    required.append(name)
                properties[name] = prop
            schema: dict[str, t.Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema
        except Exception:
            return {"type": "object", "properties": {}}

    @staticmethod
    def _map_annotation_to_json_type(ann: t.Any) -> str | None:
        try:
            origin = t.get_origin(ann) or ann
            mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                tuple: "array",
                set: "array",
                dict: "object",
            }
            # Typed containers map to base Python containers in get_origin
            return mapping.get(origin)
        except Exception:
            return None

    def to_schema(self) -> dict[str, t.Any]:
        """
        Return the function-calling schema for the wrapped tool.

        Returns:
            dict[str, Any]: Function-calling schema with name, description, parameters.
        """
        schema = self._json_schema_from_args_schema() or self._infer_schema_from_signature()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def execute(self, arguments: dict[str, t.Any]) -> dict[str, t.Any]:
        """
        Execute the wrapped tool with the provided arguments.

        Args:
            arguments (dict[str, Any]): Arguments to pass to the tool.

        Returns:
            dict[str, Any]: Normalized response dict with keys: successful, data, error.
        """
        try:
            tool = self._tool
            if hasattr(tool, "invoke"):
                result = tool.invoke(arguments)  # type: ignore[misc]
            elif hasattr(tool, "run"):
                result = tool.run(arguments)  # type: ignore[misc]
            elif hasattr(tool, "_run"):
                result = tool._run(arguments)  # type: ignore[attr-defined]
            elif callable(self._callable):
                result = self._callable(**arguments)  # type: ignore[call-arg]
            else:
                raise AttributeError("Tool does not support invoke/run/_run/callable")

            data: t.Any = result
            if not isinstance(result, str | int | float | bool | type(None) | dict | list):
                try:
                    json.dumps(result)
                except Exception:
                    data = str(result)
            return {"successful": True, "data": data, "error": None}
        except Exception as exc:
            logger.error("LangChain wrapped tool '%s' failed: %s", self.name, exc)
            return {"successful": False, "data": None, "error": str(exc)}


class LangChainAdapter:
    """
    Generic registry-based LangChain adapter.

    Notes:
        - Avoids importing heavy integrations until needed (lazy default autoload).
        - Normalizes schemas and execution results into simple dicts.
        - Allows arbitrary tool registration instead of hardcoding a tiny set.
    """

    def __init__(self, *, autoload_default_tools: bool = True) -> None:
        """
        Initialize LangChainAdapter.

        Args:
            autoload_default_tools (bool): Whether to autoload default tools if registry is empty.

        Raises:
            ImportError: If langchain-core is not installed.
        """
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChainAdapter requires 'langchain-core' and optional integrations.\n"
                "Install with: pip install 10xscale-agentflow[langchain]"
            )
        self._registry: dict[str, LangChainToolWrapper] = {}
        self._autoload = autoload_default_tools

    @staticmethod
    def is_available() -> bool:
        """
        Return True if langchain-core is importable.

        Returns:
            bool: True if langchain-core is available, False otherwise.
        """
        return HAS_LANGCHAIN

    # ------------------------
    # Discovery
    # ------------------------
    def list_tools_for_llm(self) -> list[dict[str, t.Any]]:
        """
        Return a list of function-calling formatted tool schemas.

        If registry is empty and autoload is enabled, attempt to autoload a
        couple of common tools for convenience (tavily_search, requests_get).

        Returns:
            list[dict[str, Any]]: List of tool schemas in function-calling format.
        """
        if not self._registry and self._autoload:
            self._try_autoload_defaults()

        return [wrapper.to_schema() for wrapper in self._registry.values()]

    # ------------------------
    # Execute
    # ------------------------
    def execute(self, *, name: str, arguments: dict[str, t.Any]) -> dict[str, t.Any]:
        """
        Execute a supported LangChain tool and normalize the response.

        Args:
            name (str): Name of the tool to execute.
            arguments (dict[str, Any]): Arguments for the tool.

        Returns:
            dict[str, Any]: Normalized response dict with keys: successful, data, error.
        """
        if name not in self._registry and self._autoload:
            # Late autoload attempt in case discovery wasn't called first
            self._try_autoload_defaults()

        wrapper = self._registry.get(name)
        if not wrapper:
            return {"successful": False, "data": None, "error": f"Unknown LangChain tool: {name}"}
        return wrapper.execute(arguments)

    # ------------------------
    # Internals
    # ------------------------
    def register_tool(
        self,
        tool: t.Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Register a tool instance and return the resolved name used for exposure.

        Args:
            tool (Any): Tool instance to register.
            name (str | None): Optional override for tool name.
            description (str | None): Optional override for tool description.

        Returns:
            str: The resolved name used for exposure.
        """
        wrapper = LangChainToolWrapper(tool, name=name, description=description)
        self._registry[wrapper.name] = wrapper
        return wrapper.name

    def register_tools(self, tools: list[t.Any]) -> list[str]:
        """
        Register multiple tool instances.

        Args:
            tools (list[Any]): List of tool instances to register.

        Returns:
            list[str]: List of resolved names for the registered tools.
        """
        names: list[str] = []
        for tool in tools:
            names.append(self.register_tool(tool))
        return names

    def _create_tavily_search_tool(self) -> t.Any:
        """
        Construct Tavily search tool lazily.

        Prefer the new dedicated integration `langchain_tavily.TavilySearch`.
        Fall back to the deprecated community tool if needed.

        Returns:
            Any: Tavily search tool instance.

        Raises:
            ImportError: If Tavily tool cannot be imported.
        """
        # Preferred: langchain-tavily
        try:
            mod = importlib.import_module("langchain_tavily")
            return mod.TavilySearch()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Preferred langchain_tavily import failed: %s", exc)

        # Fallback: deprecated community tool (still functional for now)
        try:
            mod = importlib.import_module("langchain_community.tools.tavily_search")
            return mod.TavilySearchResults()
        except Exception as exc:  # ImportError or runtime
            raise ImportError(
                "Tavily tool requires 'langchain-tavily' (preferred) or"
                " 'langchain-community' with 'tavily-python'.\n"
                "Install with: pip install 10xscale-agentflow[langchain]"
            ) from exc

    def _create_requests_get_tool(self) -> t.Any:
        """
        Construct RequestsGetTool lazily with a basic requests wrapper.

        Note: Requests tools require an explicit wrapper instance and, for safety,
        default to disallowing dangerous requests. Here we opt-in to allow GET
        requests by setting allow_dangerous_requests=True to make the tool usable
        in agent contexts. Consider tightening this in your application.

        Returns:
            Any: RequestsGetTool instance.

        Raises:
            ImportError: If RequestsGetTool cannot be imported.
        """
        try:
            req_tool_mod = importlib.import_module("langchain_community.tools.requests.tool")
            util_mod = importlib.import_module("langchain_community.utilities.requests")
            wrapper = util_mod.TextRequestsWrapper(headers={})  # type: ignore[attr-defined]
            return req_tool_mod.RequestsGetTool(
                requests_wrapper=wrapper,
                allow_dangerous_requests=True,
            )
        except Exception as exc:  # ImportError or runtime
            raise ImportError(
                "Requests tool requires 'langchain-community'.\n"
                "Install with: pip install 10xscale-agentflow[langchain]"
            ) from exc

    def _try_autoload_defaults(self) -> None:
        """
        Best-effort autoload of a couple of common tools.

        This keeps prior behavior available while allowing users to register
        arbitrary tools. Failures are logged but non-fatal.

        Returns:
            None
        """
        # Tavily search
        try:
            tavily = self._create_tavily_search_tool()
            self.register_tool(tavily, name="tavily_search")
        except Exception as exc:
            logger.debug("Skipping Tavily autoload: %s", exc)

        # Requests GET
        try:
            rget = self._create_requests_get_tool()
            self.register_tool(rget, name="requests_get")
        except Exception as exc:
            logger.debug("Skipping requests_get autoload: %s", exc)

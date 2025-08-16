"""Tool node utilities.

Provides a ToolNode that inspects callables and provides JSON-schema-like
descriptions suitable for function-calling LLMs and a simple execute API.
"""

from __future__ import annotations

import inspect
import typing as t


if t.TYPE_CHECKING:
    from pyagenity.graph.checkpointer import BaseCheckpointer, BaseStore
    from pyagenity.graph.state import AgentState

from pyagenity.graph.utils.callable_utils import call_sync_or_async
from pyagenity.graph.utils.injectable import get_injectable_param_name, is_injectable_type


class ToolNode:
    """Registry for callables that exports function specs and executes them."""

    def __init__(self, functions: t.Iterable[t.Callable]):
        self._funcs: dict[str, t.Callable] = {}
        for fn in functions:
            if not callable(fn):
                raise TypeError("ToolNode only accepts callables")
            self._funcs[fn.__name__] = fn

    def all_tools(self) -> list[dict]:
        """Return function descriptions for all registered callables."""

        tools: list[dict] = []
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

        return tools

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
    ) -> t.Any:
        """Execute the callable registered under `name` with `args` kwargs.

        Additional injectable parameters:
        - tool_call_id: ID of the tool call (can be injected into function if needed)
        - state: Current agent state (can be injected into function if needed)
        - checkpointer: Checkpointer instance (can be injected into function if needed)
        - store: Store instance (can be injected into function if needed)
        - dependency_container: Container with custom dependencies
        """

        if name not in self._funcs:
            raise KeyError(f"Function '{name}' is not registered in ToolNode")

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

        kwargs: dict = {}
        for p_name, p in sig.parameters.items():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # Check if this parameter should be injected based on type annotation
            annotation = p.annotation if p.annotation is not inspect._empty else None
            if annotation and is_injectable_type(annotation):
                injectable_param_name = get_injectable_param_name(annotation)

                if injectable_param_name == "dependency":
                    # For InjectDep, use the parameter name to look up the dependency
                    if dependency_container and dependency_container.has(p_name):
                        kwargs[p_name] = dependency_container.get(p_name)
                    elif p.default is inspect._empty:
                        # Required dependency not found
                        raise TypeError(f"Required dependency '{p_name}' not found in container")
                    # If default exists and dependency not found, don't inject (use default)
                elif injectable_param_name and injectable_param_name in injectable_params:
                    injectable_value = injectable_params[injectable_param_name]
                    if injectable_value is not None:
                        kwargs[p_name] = injectable_value
                continue

            # First try to use from args (function arguments)
            if p_name in args:
                kwargs[p_name] = args[p_name]
            # Then try injectable parameters (legacy support for direct parameter names)
            elif p_name in injectable_params and injectable_params[p_name] is not None:
                kwargs[p_name] = injectable_params[p_name]
            # Try dependency container for non-annotated parameters
            elif dependency_container and dependency_container.has(p_name):
                kwargs[p_name] = dependency_container.get(p_name)
            # Finally use default if available
            elif p.default is not inspect._empty:
                # omit to use default
                pass
            else:
                raise TypeError(f"Missing required parameter '{p_name}' for function '{name}'")

        return await call_sync_or_async(fn, **kwargs)

    @staticmethod
    def _annotation_to_schema(annotation: t.Any, default: t.Any) -> dict:
        """Map simple Python annotations to JSON-schema-like dicts.

        Supports basic primitives, list[...] and typing.Literal for enums.
        Falls back to string when unknown.
        """

        origin = getattr(annotation, "__origin__", None)

        # Handle Optional[...] / Union[..., None]
        args = getattr(annotation, "__args__", None)
        if args and any(a is type(None) for a in args):
            # pick the non-None arg and map that
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return ToolNode._annotation_to_schema(non_none[0], default)

        if annotation is str:
            schema = {"type": "string"}
        elif annotation is int:
            schema = {"type": "integer"}
        elif annotation is float:
            schema = {"type": "number"}
        elif annotation is bool:
            schema = {"type": "boolean"}
        elif origin is list:
            item_type = getattr(annotation, "__args__", (str,))[0]
            item_schema = ToolNode._annotation_to_schema(item_type, None)
            schema = {"type": "array", "items": item_schema}
        else:
            Literal = getattr(t, "Literal", None)
            if Literal is not None and origin is Literal:
                literals = list(getattr(annotation, "__args__", ()))
                if all(isinstance(literal, str) for literal in literals):
                    schema = {"type": "string", "enum": literals}
                else:
                    schema = {"enum": literals}
            else:
                schema = {"type": "string"}

        if default is not inspect._empty:
            schema["default"] = default

        return schema

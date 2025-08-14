"""Tool node utilities.

Provides a ToolNode that inspects callables and provides JSON-schema-like
descriptions suitable for function-calling LLMs and a simple execute API.
"""

from __future__ import annotations

import inspect
import typing as t


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

        return tools

    def execute(self, name: str, args: dict) -> t.Any:
        """Execute the callable registered under `name` with `args` kwargs."""

        if name not in self._funcs:
            raise KeyError(f"Function '{name}' is not registered in ToolNode")

        fn = self._funcs[name]
        sig = inspect.signature(fn)

        kwargs: dict = {}
        for p_name, p in sig.parameters.items():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if p_name in args:
                kwargs[p_name] = args[p_name]
            elif p.default is not inspect._empty:
                # omit to use default
                pass
            else:
                raise TypeError(f"Missing required parameter '{p_name}' for function '{name}'")

        return fn(**kwargs)

    def get_callable(self, name: str) -> t.Callable:
        return self._funcs[name]

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
                if all(isinstance(l, str) for l in literals):
                    schema = {"type": "string", "enum": literals}
                else:
                    schema = {"enum": literals}
            else:
                schema = {"type": "string"}

        if default is not inspect._empty:
            schema["default"] = default

        return schema

"""Schema utilities and local tool description building for ToolNode."""

from __future__ import annotations

import inspect
import typing as t

from .constants import INJECTABLE_PARAMS


class SchemaMixin:
    """Provides schema helpers and local tool description building."""

    _funcs: dict[str, t.Callable]

    @staticmethod
    def _handle_optional_annotation(annotation: t.Any, default: t.Any) -> dict | None:
        args = getattr(annotation, "__args__", None)
        if args and any(a is type(None) for a in args):
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return SchemaMixin._annotation_to_schema(non_none[0], default)
        return None

    @staticmethod
    def _handle_complex_annotation(annotation: t.Any) -> dict:
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            item_type = getattr(annotation, "__args__", (str,))[0]
            item_schema = SchemaMixin._annotation_to_schema(item_type, None)
            return {"type": "array", "items": item_schema}

        Literal = getattr(t, "Literal", None)
        if Literal is not None and origin is Literal:
            literals = list(getattr(annotation, "__args__", ()))
            if all(isinstance(literal, str) for literal in literals):
                return {"type": "string", "enum": literals}
            return {"enum": literals}

        return {"type": "string"}

    @staticmethod
    def _annotation_to_schema(annotation: t.Any, default: t.Any) -> dict:
        schema = SchemaMixin._handle_optional_annotation(annotation, default)
        if schema:
            return schema

        primitive_mappings = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }

        if annotation in primitive_mappings:
            schema = primitive_mappings[annotation]
        else:
            schema = SchemaMixin._handle_complex_annotation(annotation)

        if default is not inspect._empty:
            schema["default"] = default

        return schema

    def get_local_tool(self) -> list[dict]:
        tools: list[dict] = []
        for name, fn in self._funcs.items():
            sig = inspect.signature(fn)
            params_schema: dict = {"type": "object", "properties": {}, "required": []}

            for p_name, p in sig.parameters.items():
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                if p_name in INJECTABLE_PARAMS:
                    continue

                annotation = p.annotation if p.annotation is not inspect._empty else str
                prop = SchemaMixin._annotation_to_schema(annotation, p.default)
                params_schema["properties"][p_name] = prop

                if p.default is inspect._empty:
                    params_schema["required"].append(p_name)

            if not params_schema["required"]:
                params_schema.pop("required")

            description = inspect.getdoc(fn) or "No description provided."

            # provider = getattr(fn, "_py_tool_provider", None)
            # tags = getattr(fn, "_py_tool_tags", None)
            # capabilities = getattr(fn, "_py_tool_capabilities", None)

            entry = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params_schema,
                },
            }
            # meta: dict[str, t.Any] = {}
            # if provider:
            #     meta["provider"] = provider
            # if tags:
            #     meta["tags"] = tags
            # if capabilities:
            #     meta["capabilities"] = capabilities
            # if meta:
            #     entry["x-pyagenity"] = meta

            tools.append(entry)

        return tools

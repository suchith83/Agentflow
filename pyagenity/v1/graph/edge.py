# from __future__ import annotations

# from collections.abc import Callable
# from typing import Any


# class Edge:
#     """Represents a directed conditional edge between two nodes.

#     condition receives the shared state (dict) and returns bool.
#     """

#     def __init__(
#         self,
#         source: str,
#         target: str,
#         condition: Callable[[dict[str, Any]], bool] | None = None,
#     ):
#         self.source = source
#         self.target = target
#         self.condition = condition

#     def is_triggered(self, context: dict[str, Any]) -> bool:
#         if self.condition is None:
#             return True
#         try:
#             return bool(self.condition(context))
#         except Exception:
#             # Defensive: failing conditions should not propagate exceptions
#             return False

#     def __repr__(self) -> str:  # pragma: no cover - simple repr
#         cond = (
#             self.condition.__name__
#             if self.condition and hasattr(self.condition, "__name__")
#             else "<lambda>"
#             if self.condition
#             else "<always>"
#         )
#         return f"Edge({self.source} -> {self.target}, condition={cond})"

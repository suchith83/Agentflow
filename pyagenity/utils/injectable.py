"""Injectable parameter types for tool functions.

These types indicate that parameters should be automatically injected by the framework
and should not be included in the LLM tool specification.
"""

from typing import TYPE_CHECKING, Generic, TypeVar


if TYPE_CHECKING:
    pass

T = TypeVar("T")


class _InjectableType(Generic[T]):
    """Base class for injectable parameter types.

    This is a generic class that allows specifying the exact type
    that will be injected, providing proper IDE type hints.
    """

    def __init__(self, annotation: type | None = None):
        self.annotation = annotation

    def __class_getitem__(cls, item):
        # This creates a new generic type with the specified parameter
        # e.g., InjectState[MyState] creates a type that IDEs understand
        return type(
            f"{cls.__name__}[{getattr(item, '__name__', str(item))}]",
            (cls,),
            {
                "__origin__": cls,
                "__args__": (item,) if not isinstance(item, tuple) else item,
                "__parameters__": (),
            },
        )


class InjectToolCallID(_InjectableType[str]):
    """Type annotation indicating that tool_call_id should be injected.

    Usage:
        tool_call_id: InjectToolCallID = None
        # or with explicit type:
        tool_call_id: InjectToolCallID[str] = None
    """


class InjectState(_InjectableType["AgentState"]):
    """Type annotation indicating that agent state should be injected.

    Usage:
        state: InjectState = None
        # or with custom state type:
        state: InjectState[MyCustomState] = None
    """


class InjectCheckpointer(_InjectableType["BaseCheckpointer"]):
    """Type annotation indicating that checkpointer should be injected.

    Usage:
        checkpointer: InjectCheckpointer = None
        # or with custom checkpointer type:
        checkpointer: InjectCheckpointer[MyCheckpointer] = None
    """


class InjectStore(_InjectableType["BaseStore"]):
    """Type annotation indicating that store should be injected.

    Usage:
        store: InjectStore = None
        # or with custom store type:
        store: InjectStore[MyStore] = None
    """


class InjectConfig(_InjectableType[dict]):
    """Type annotation indicating that config should be injected.

    Usage:
        config: InjectConfig = None
        # or with typed config:
        config: InjectConfig[MyConfigDict] = None
    """


class InjectDep(_InjectableType[T]):
    """Type annotation indicating that a custom dependency should be injected.

    The dependency name is automatically derived from the parameter name.

    Usage:
        def my_node(database: InjectDep[Database] = None):
            # 'database' dependency will be looked up and injected
            pass

        # or with explicit type:
        def my_node(logger: InjectDep[Logger] = None):
            # 'logger' dependency will be looked up and injected
            pass
    """


def is_injectable_type(annotation) -> bool:
    """Check if an annotation is an injectable type."""
    # Handle both direct class and generic instances
    if hasattr(annotation, "__origin__"):
        # For generic types like InjectState[MyState]
        origin = getattr(annotation, "__origin__", None)
        return isinstance(origin, type) and issubclass(origin, _InjectableType)

    # For direct class usage like InjectState
    try:
        return isinstance(annotation, type) and issubclass(annotation, _InjectableType)
    except TypeError:
        return False


def get_injectable_param_name(annotation) -> str | None:
    """Get the injectable parameter name for a given annotation."""
    # Handle both direct class and generic instances
    original_annotation = annotation
    if hasattr(annotation, "__origin__"):
        annotation = annotation.__origin__

    # Mapping of annotation types to parameter names
    injectable_mappings = {
        InjectToolCallID: "tool_call_id",
        InjectState: "state",
        InjectCheckpointer: "checkpointer",
        InjectStore: "store",
        InjectConfig: "config",
        InjectDep: "dependency",
    }

    # Check direct annotation match first
    for injectable_type, param_name in injectable_mappings.items():
        if annotation is injectable_type:
            return param_name

    # Check if it's a subclass for backwards compatibility
    if isinstance(original_annotation, type):
        for injectable_type, param_name in injectable_mappings.items():
            try:
                if issubclass(original_annotation, injectable_type):
                    return param_name
            except TypeError:
                # issubclass may fail for some types
                continue

    return None

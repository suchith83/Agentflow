# from injectq import InjectQ


# data = {"key1": "value1", "key2": "value2", "key3": "value3"}


# def generate_name(messages: list[str]) -> str:
#     return data.get(messages[0], "default_value")


# if __name__ == "__main__":
#     injector = InjectQ()
#     injector.bind_factory("data_store", generate_name)

#     result = injector.invoke("data_store", ["key2"])
#     print(result)


import importlib
import json
from typing import Type, TypeVar

from agentflow.state import AgentState
from agentflow.state.message import Message


StateT = TypeVar("StateT", bound=AgentState)


def _get_full_class_path(obj: object) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _import_class_from_path(path: str) -> Type[AgentState]:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def save_state(state: AgentState) -> str:
    """Serialize with class path."""
    data = state.model_dump()
    data["__class_path__"] = _get_full_class_path(state)
    return json.dumps(data)


def load_state(json_data: str) -> AgentState:
    """Deserialize dynamically using class path."""
    data = json.loads(json_data)
    class_path = data.pop("__class_path__", None)
    if not class_path:
        raise ValueError("Missing '__class_path__' in JSON data")

    cls = _import_class_from_path(class_path)
    return cls.model_validate(data)


class CustomState(AgentState):
    jd_id: str
    user_name: str


state = CustomState(
    jd_id="12345",
    user_name="test_user",
    context=[Message.text_message("Hi"), Message.text_message("How are you?")],
    context_summary="Summary",
)

json_data = save_state(state)
print("Serialized:", json_data)

restored = load_state(json_data)
print("Restored:", restored)
print("Type:", type(restored))

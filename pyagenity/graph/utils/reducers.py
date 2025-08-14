from .message import Message


def add_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer that adds messages to the list."""
    return left + right


def replace_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer that replaces the entire message list."""
    return right


def append_items(left: list, right: list) -> list:
    """Generic reducer that appends items to a list."""
    return left + right


def replace_value(left, right):
    """Generic reducer that replaces a value."""
    return right

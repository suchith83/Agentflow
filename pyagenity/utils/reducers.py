from .message import Message


def add_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer that adds messages to the list."""
    left_ids = {msg.message_id for msg in left}
    right = [msg for msg in right if msg.message_id not in left_ids]
    return left + right


def replace_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer that replaces the entire message list."""
    return right


def append_items(left: list, right: list) -> list:
    """Generic reducer that appends items to a list."""
    left_ids = {item.id for item in left}
    right = [item for item in right if item.id not in left_ids]
    return left + right


def replace_value(left, right):
    """Generic reducer that replaces a value."""
    return right

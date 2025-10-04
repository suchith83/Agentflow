"""
Reducer utilities for merging and replacing lists and values in agent state.

This module provides generic and message-specific reducers for combining lists,
replacing values, and appending items while avoiding duplicates.

Functions:
    add_messages: Adds messages to a list, avoiding duplicates by message_id.
    replace_messages: Replaces the entire message list.
    append_items: Appends items to a list, avoiding duplicates by id.
    replace_value: Replaces a value with another.
"""

from .message import Message


def add_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """
    Adds messages to the list, avoiding duplicates by message_id.

    Args:
        left (list[Message]): Existing list of messages.
        right (list[Message]): New messages to add.

    Returns:
        list[Message]: Combined list with unique messages.

    Example:
        >>> add_messages([msg1], [msg2, msg1])
        [msg1, msg2]
    """
    left_ids = {msg.message_id for msg in left}
    right = [msg for msg in right if msg.message_id not in left_ids]
    return left + right


def replace_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """
    Replaces the entire message list with a new one.

    Args:
        left (list[Message]): Existing list of messages (ignored).
        right (list[Message]): New list of messages.

    Returns:
        list[Message]: The new message list.

    Example:
        >>> replace_messages([msg1], [msg2])
        [msg2]
    """
    return right


def append_items(left: list, right: list) -> list:
    """
    Appends items to a list, avoiding duplicates by item.id.

    Args:
        left (list): Existing list of items (must have .id attribute).
        right (list): New items to add.

    Returns:
        list: Combined list with unique items.

    Example:
        >>> append_items([item1], [item2, item1])
        [item1, item2]
    """
    left_ids = {item.id for item in left}
    right = [item for item in right if item.id not in left_ids]
    return left + right


def replace_value(left, right):
    """
    Replaces a value with another.

    Args:
        left: Existing value (ignored).
        right: New value to use.

    Returns:
        Any: The new value.

    Example:
        >>> replace_value(1, 2)
        2
    """
    return right

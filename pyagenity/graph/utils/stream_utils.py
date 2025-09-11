from litellm.types.utils import ModelResponse

from pyagenity.utils import message
from pyagenity.utils.command import Command


def check_non_streaming(result) -> bool:
    # check list, dict, str, Command, Message, Model Response or not
    if isinstance(result, (list, dict, str)):
        return True

    if isinstance(result, ModelResponse):
        return True

    if isinstance(result, Command):
        return True

    return bool(isinstance(result, message))

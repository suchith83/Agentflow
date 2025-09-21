from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message


def check_non_streaming(result) -> bool:
    # check list, dict, str, Command, Message, Model Response or not
    if isinstance(result, list | dict | str):
        return True

    if isinstance(result, Message):
        return True

    if isinstance(result, AgentState):
        return True

    if isinstance(result, dict) and "choices" in result:
        return True

    return bool(isinstance(result, Message))

from collections.abc import Callable
from typing import TypeVar

from injectq import InjectQ

from agentflow.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.graph.state_graph import StateGraph
from agentflow.publisher.base_publisher import BasePublisher
from agentflow.state.agent_state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


def _should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done (AI will make final response)
    if last_message.role == "tool" and last_message is not None:
        return "MAIN"

    # Default to END for other cases
    return END


class ReactAgent[StateT: AgentState]:
    def __init__(
        self,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
    ):
        self._graph = StateGraph[StateT](
            state=state,
            context_manager=context_manager,
            publisher=publisher,
            id_generator=id_generator,
            container=container,
        )

    def compile(
        self,
        main_node: tuple[Callable, str] | Callable,
        tool_node: tuple[Callable, str] | Callable,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Determine main node function and name
        if isinstance(main_node, tuple):
            main_func, main_name = main_node
            if not callable(main_func):
                raise ValueError("main_node[0] must be a callable function")
        else:
            main_func = main_node
            main_name = "MAIN"
            if not callable(main_func):
                raise ValueError("main_node must be a callable function")

        # Determine tool node function and name
        if isinstance(tool_node, tuple):
            tool_func, tool_name = tool_node
            # Accept both callable functions and ToolNode instances
            if not callable(tool_func) and not hasattr(tool_func, "invoke"):
                raise ValueError("tool_node[0] must be a callable function or ToolNode")
        else:
            tool_func = tool_node
            tool_name = "TOOL"
            # Accept both callable functions and ToolNode instances
            # ToolNode instances have an 'invoke' method but are not callable
            if not callable(tool_func) and not hasattr(tool_func, "invoke"):
                raise ValueError("tool_node must be a callable function or ToolNode instance")

        self._graph.add_node(main_name, main_func)
        self._graph.add_node(tool_name, tool_func)

        # Now create edges
        self._graph.add_conditional_edges(
            main_name,
            _should_use_tools,
            {tool_name: tool_name, END: END},
        )

        self._graph.add_edge(tool_name, main_name)
        self._graph.set_entry_point(main_name)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )

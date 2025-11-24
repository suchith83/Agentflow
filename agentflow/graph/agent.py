"""Agent class for simplified LLM interactions in TAF graphs.

This module provides a high-level Agent wrapper that acts as a smart node function,
handling common boilerplate for LLM interactions including message conversion,
tool handling, and optional learning/RAG capabilities.

The Agent class is designed to be used as a node within a StateGraph, not as
a replacement for the graph itself.
"""

import logging
from collections.abc import Callable
from typing import Any

from injectq import Inject, InjectQ

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph.tool_node import ToolNode
from agentflow.state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.state.message import Message
from agentflow.store.base_store import BaseStore
from agentflow.utils.converter import convert_messages


try:
    from litellm import acompletion

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    acompletion = None  # type: ignore[assignment]


logger = logging.getLogger("agentflow.agent")

# Constants
CONTENT_PREVIEW_LENGTH = 200


class Agent:
    """A smart node function wrapper for LLM interactions.

    This class handles common boilerplate for agent implementations including:
    - Automatic message conversion
    - LLM calls via LiteLLM
    - Tool handling with conditional logic
    - Optional learning/RAG capabilities
    - Response conversion

    The Agent is designed to be used as a node within a StateGraph, providing
    a high-level interface while maintaining full graph flexibility.

    Example:
        ```python
        # Create an agent node
        agent = Agent(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            tools=[weather_tool],
            learning=True,
        )

        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)  # Agent acts as a node function
        graph.add_node("TOOL", agent.get_tool_node())
        # ... setup edges
        ```

    Attributes:
        model: LiteLLM model identifier
        system_prompt: System prompt string or list of message dicts
        tools: List of tool functions or ToolNode instance
        learning: Whether to enable automatic learning/RAG
        store: Store instance for learning (required if learning=True)
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate
        llm_kwargs: Additional LiteLLM parameters
    """

    def __init__(
        self,
        model: str,
        system_prompt: list[dict[str, Any]],
        tools: list[Callable] | ToolNode | None = None,
        tool_node_name: str | None = None,
        extra_messages: list[Message] | None = None,
        learning: bool = False,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        **llm_kwargs,
    ):
        """Initialize an Agent node.

        Args:
            model: LiteLLM model string (e.g., "gpt-4", "gemini/gemini-2.0-flash").
                See https://docs.litellm.ai/docs/providers for supported models.
            system_prompt: System prompt string or list of system message dicts.
                Can include cache control and other provider-specific options.
            tools: List of tool functions, ToolNode instance, or None.
                If list is provided, will be converted to ToolNode internally.
            tool_node_name: Name of the existing ToolNode. You can sent list of tools
                or provide ToolNode instance via `tools` parameter instead.
            extra_messages: Additional messages to include in every interaction.
            learning: Enable automatic RAG-based learning. When True, the agent
                will retrieve relevant past interactions before responding and
                automatically store new Q&A pairs. Requires store parameter.
            store: BaseStore instance for learning (required if learning=True).
                Used for both retrieval and storage of interactions.
            temperature: LLM temperature parameter (0.0-2.0). Lower values make
                output more deterministic, higher values more creative.
            max_tokens: Maximum tokens to generate. If None, uses model default.
            **llm_kwargs: Additional LiteLLM parameters (top_p, top_k,
                frequency_penalty, presence_penalty, etc.).

        Raises:
            ValueError: If learning=True but store=None.

        Example:
            ```python
            agent = Agent(
                model="gpt-4",
                system_prompt="You are a helpful assistant",
                tools=[weather_tool, calculator],
                learning=True,
                store=my_store,
                temperature=0.8,
                max_tokens=1000,
                top_p=0.9,
            )
            ```
        """
        logger.debug(f"Initializing Agent with model={model}, learning={learning}")

        if not HAS_LITELLM:
            raise ImportError(
                "litellm is required for Agent class. "
                "Install it with: pip install 10xscale-agentflow[litellm]"
            )

        self.model = model
        self.system_prompt = system_prompt
        self.extra_messages = extra_messages
        self.tools = tools
        self.learning = learning
        self.llm_kwargs = llm_kwargs
        self.trim_context = trim_context
        self.tools_tags = tools_tags
        self.tool_node_name = tool_node_name

        # Internal setup
        self._tool_node = self._setup_tools()

        logger.info(
            f"Agent initialized: model={model}, has_tools={self._tool_node is not None}, "
            f"learning={learning}"
        )

    def _setup_tools(self) -> ToolNode | None:
        """Convert tools to ToolNode if needed.

        Returns:
            ToolNode instance or None if no tools provided.
        """
        if self.tools is None:
            logger.debug("No tools provided")
            return None

        if isinstance(self.tools, ToolNode):
            logger.debug("Tools already a ToolNode instance")
            return self.tools

        logger.debug(f"Converting {len(self.tools)} tool functions to ToolNode")
        return ToolNode(self.tools)

    async def _trim_context(
        self,
        state: AgentState,
        context_manager: BaseContextManager | None = Inject[BaseContextManager],
    ) -> AgentState:
        """Trim context using the provided context manager.

        Args:
            state: Current agent state.
            context_manager: Context manager for trimming.

        Returns:
            Trimmed agent state.
        """
        if self.trim_context:
            if context_manager is None:
                logger.warning("trim_context is enabled but no context manager is available")
                return state

            new_state = await context_manager.trim_context(state)
            logger.debug("Context trimmed using context manager")
            return new_state

        logger.debug("Context trimming not enabled")
        return state

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ):
        container = InjectQ.get_instance()

        # check store
        store: BaseStore | None = container.try_get(BaseStore)

        if self.learning and store is None:
            logger.warning(
                "Learning is enabled but no BaseStore instance found in InjectQ container."
                " Ignoring learning."
            )

        # trim context if enabled
        state = await self._trim_context(state)

        # convert messages for LLM
        messages = convert_messages(
            system_prompts=self.system_prompt,
            state=state,
            extra_messages=self.extra_messages or [],
        )

        # check is is_stream enabled or not
        is_stream = config.get("is_stream", False)

        # If tool results just came in, make final response without tools
        if state.context and len(state.context) > 0 and state.context[-1].role == "tool":
            # Make final response without tools since we just got tool results
            response = acompletion(
                model=self.model,
                messages=messages,
                stream=is_stream,
                **self.llm_kwargs,
            )  # type: ignore
        else:
            # Regular response with tools available
            tools = []
            if self._tool_node:
                tools = await self._tool_node.all_tools(tags=self.tools_tags)

            # check form tool node name
            if self.tool_node_name:
                try:
                    node = container.call_factory(self.tool_node_name)
                except KeyError:
                    logger.warning(
                        f"ToolNode with name '{self.tool_node_name}' not found in InjectQ registry."
                    )
                    node = None

                if node and isinstance(node, ToolNode):
                    tools = await node.all_tools(tags=self.tools_tags)

            response = acompletion(
                model=self.model,
                messages=messages,
                stream=is_stream,
                tools=tools,
                **self.llm_kwargs,
            )  # type: ignore

        return ModelResponseConverter(
            response,
            converter="litellm",
        )

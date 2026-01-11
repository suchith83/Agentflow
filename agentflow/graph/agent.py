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
from agentflow.graph.base_agent import BaseAgent
from agentflow.graph.tool_node import ToolNode
from agentflow.state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.state.message import Message

# from agentflow.store.base_store import BaseStore
from agentflow.utils.converter import convert_messages


logger = logging.getLogger("agentflow.agent")

# Constants
CONTENT_PREVIEW_LENGTH = 200


class Agent(BaseAgent):
    """A smart node function wrapper for LLM interactions.

    This class handles common boilerplate for agent implementations including:
    - Automatic message conversion
    - LLM calls via native provider SDKs (OpenAI, Anthropic, Google)
    - Tool handling with conditional logic
    - Optional learning/RAG capabilities
    - Response conversion

    The Agent is designed to be used as a node within a StateGraph, providing
    a high-level interface while maintaining full graph flexibility.

    Example:
        ```python
        # Create an agent node with OpenAI
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            system_prompt="You are a helpful assistant",
            tools=[weather_tool],
        )

        # Or with Anthropic
        agent = Agent(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            system_prompt="You are a helpful assistant",
        )

        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)  # Agent acts as a node function
        graph.add_node("TOOL", agent.get_tool_node())
        # ... setup edges
        ```

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        provider: Provider name ("openai", "anthropic", "google")
        system_prompt: System prompt string or list of message dicts
        tools: List of tool functions or ToolNode instance
        client: Optional custom client instance (escape hatch for power users)
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate
        llm_kwargs: Additional provider-specific parameters
    """

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tools: list[Callable] | ToolNode | None = None,
        tool_node_name: str | None = None,
        extra_messages: list[Message] | None = None,
        client: Any = None,  # Escape hatch: allow custom client
        base_url: str | None = None,  # For OpenAI-compatible APIs (ollama, vllm, etc.)
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        **llm_kwargs,
    ):
        """Initialize an Agent node.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gemini-2.0-flash-exp").
            provider: Provider name ("openai", "google"). If None, will auto-detect from model.
            system_prompt: System prompt as list of message dicts.
            tools: List of tool functions, ToolNode instance, or None.
                If list is provided, will be converted to ToolNode internally.
            tool_node_name: Name of the existing ToolNode. You can send list of tools
                or provide ToolNode instance via `tools` parameter instead.
            extra_messages: Additional messages to include in every interaction.
            client: Optional custom client instance (escape hatch). If provided, provider/model are ignored.
            base_url: Optional base URL for OpenAI-compatible APIs (ollama, vllm, openrouter, deepseek, etc.).
            trim_context: Whether to trim context using context manager.
            tools_tags: Optional tags to filter tools.
            **llm_kwargs: Additional provider-specific parameters (temperature, max_tokens, top_p, etc.).

        Raises:
            ImportError: If required provider SDK is not installed.
            ValueError: If provider cannot be determined.

        Example:
            ```python
            # OpenAI agent
            agent = Agent(
                model="gpt-4o",
                provider="openai",
                system_prompt="You are a helpful assistant",
                tools=[weather_tool, calculator],
                temperature=0.8,
                max_tokens=1000,
            )

            # Google Gemini agent (auto-detect provider from model)
            agent = Agent(
                model="gemini-2.0-flash-exp",
                system_prompt="You are a helpful assistant",
            )

            # Ollama via OpenAI-compatible API
            agent = Agent(
                model="llama3",
                provider="openai",
                base_url="http://localhost:11434/v1",
                system_prompt="You are a helpful assistant",
            )

            # OpenRouter
            agent = Agent(
                model="anthropic/claude-3.5-sonnet",
                provider="openai",
                base_url="https://openrouter.ai/api/v1",
                system_prompt="You are a helpful assistant",
            )

            # Custom client (escape hatch)
            from openai import AsyncOpenAI
            custom_client = AsyncOpenAI(api_key="...", base_url="https://proxy.com")
            agent = Agent(
                model="gpt-4o",
                client=custom_client,
                system_prompt="...",
            )
            ```
        """
        # Call parent constructor
        super().__init__(model=model, system_prompt=system_prompt or [], tools=tools, **llm_kwargs)

        # Determine provider
        if client is not None:
            # User provided custom client - detect provider from client type
            self.provider = self._detect_provider_from_client(client)
            self.client = client
            self.base_url = None
            logger.debug(f"Using custom client for provider: {self.provider}")
        elif provider is not None:
            self.provider = provider.lower()
            self.base_url = base_url
            self.client = self._create_client(self.provider, model, base_url)
        else:
            # Auto-detect provider from model name
            self.provider = self._detect_provider_from_model(model)
            self.base_url = base_url
            self.client = self._create_client(self.provider, model, base_url)

        self.extra_messages = extra_messages
        self.tools = tools
        self.llm_kwargs = llm_kwargs
        self.trim_context = trim_context
        self.tools_tags = tools_tags
        self.tool_node_name = tool_node_name

        # Internal setup
        self._tool_node = self._setup_tools()

        logger.info(
            f"Agent initialized: model={model}, provider={self.provider}, "
            f"has_tools={self._tool_node is not None}"
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

    def _detect_provider_from_model(self, model: str) -> str:
        """Auto-detect provider from model name.

        Args:
            model: Model identifier string

        Returns:
            Provider name ("openai", "google")

        Raises:
            ValueError: If provider cannot be determined.
        """
        model_lower = model.lower()

        if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
            return "openai"
        elif model_lower.startswith("gemini-"):
            return "google"
        else:
            # Default to openai for unknown models (works with ollama, vllm, etc.)
            logger.info(
                f"Could not auto-detect provider for model '{model}'. "
                "Defaulting to 'openai'. If using a different provider, specify explicitly."
            )
            return "openai"

    def _detect_provider_from_client(self, client: Any) -> str:
        """Detect provider from client instance type.

        Args:
            client: Client instance

        Returns:
            Provider name
        """
        client_type = type(client).__name__
        module = type(client).__module__

        if "openai" in module or "OpenAI" in client_type:
            return "openai"
        elif "google" in module or "genai" in module or "Google" in client_type or "AsyncClient" in client_type:
            return "google"
        else:
            logger.warning(
                f"Could not detect provider from client type {client_type}. "
                "Defaulting to 'openai'."
            )
            return "openai"

    def _create_client(self, provider: str, model: str, base_url: str | None = None) -> Any:
        """Create a client instance for the specified provider.

        Args:
            provider: Provider name ("openai", "google")
            model: Model identifier
            base_url: Optional base URL for OpenAI-compatible APIs

        Returns:
            Client instance

        Raises:
            ImportError: If required SDK is not installed.
        """
        if provider == "openai":
            try:
                from openai import AsyncOpenAI

                client_kwargs = {}
                if base_url:
                    client_kwargs["base_url"] = base_url
                    logger.info(f"Using custom base_url for OpenAI client: {base_url}")

                return AsyncOpenAI(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "openai SDK is required for OpenAI provider. "
                    "Install it with: pip install 10xscale-agentflow[openai]"
                )

        elif provider == "google":
            try:
                from google import genai
                import os

                # Get API key from environment
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set "
                        "for Google provider"
                    )

                # Use AsyncClient for async operations
                return genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "google-genai SDK is required for Google provider. "
                    "Install it with: pip install 10xscale-agentflow[google-genai]"
                )

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Supported providers: 'openai', 'google'"
            )

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM using the appropriate native SDK.

        This method uses duck typing to call the correct SDK method based on
        the provider. The converter will handle provider-specific response formats.

        Args:
            messages: List of message dicts for the LLM
            tools: Optional list of tool specifications
            stream: Whether to stream the response
            **kwargs: Additional LLM call parameters

        Returns:
            Provider-specific response object
        """
        call_kwargs = {**self.llm_kwargs, **kwargs}

        # OpenAI SDK (also works with OpenAI-compatible APIs via base_url)
        if self.provider == "openai":
            if tools:
                call_kwargs["tools"] = tools

            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **call_kwargs,
            )

        # Google GenAI SDK with AsyncClient
        elif self.provider == "google":
            from google.genai import types

            # Google GenAI uses different message format
            # Extract system instruction
            system_instruction = None
            google_contents = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    if system_instruction is None:
                        system_instruction = str(content)
                    else:
                        system_instruction += "\n" + str(content)
                elif role in ["user", "assistant"]:
                    google_contents.append(str(content))

            # Build config
            config_kwargs = {}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            # Add other LLM kwargs
            if "temperature" in call_kwargs:
                config_kwargs["temperature"] = call_kwargs.pop("temperature")
            if "max_tokens" in call_kwargs or "max_output_tokens" in call_kwargs:
                config_kwargs["max_output_tokens"] = call_kwargs.pop(
                    "max_tokens", call_kwargs.pop("max_output_tokens", None)
                )

            if tools:
                # Convert tools to Google GenAI format
                google_tools = []
                for tool in tools:
                    # Google GenAI expects function definitions
                    if isinstance(tool, dict) and "function" in tool:
                        google_tools.append(tool)
                if google_tools:
                    config_kwargs["tools"] = google_tools

            config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

            # Use AsyncClient methods (native async, no wrapping needed!)
            if stream:
                # AsyncClient has async generator methods
                return await self.client.io.models.generate_content_stream(
                    model=self.model,
                    contents=google_contents,
                    config=config,
                )
            else:
                # AsyncClient has async methods
                return await self.client.io.models.generate_content(
                    model=self.model,
                    contents=google_contents,
                    config=config,
                )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ):
        container = InjectQ.get_instance()

        # check store
        # store: BaseStore | None = container.try_get(BaseStore)

        # if self.learning and store is None:
        #     logger.warning(
        #         "Learning is enabled but no BaseStore instance found in InjectQ container."
        #         " Ignoring learning."
        #     )

        # if store and self.learning:
        #     # retrieve relevant past interactions
        #     state = await store.asearch(state)

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
            response = await self._call_llm(
                messages=messages,
                stream=is_stream,
            )
        else:
            # Regular response with tools available
            tools = []
            if self._tool_node:
                tools = await self._tool_node.all_tools(tags=self.tools_tags)

            # check form tool node name
            if self.tool_node_name:
                try:
                    node = container.call_factory("get_node", self.tool_node_name)
                except KeyError:
                    logger.warning(
                        f"ToolNode with name '{self.tool_node_name}' not found in InjectQ registry."
                    )
                    node = None

                if node and isinstance(node.func, ToolNode):
                    tools = await node.func.all_tools(tags=self.tools_tags)

            response = await self._call_llm(
                messages=messages,
                tools=tools if tools else None,
                stream=is_stream,
            )

        # Use provider-specific converter
        return ModelResponseConverter(
            response,
            converter=self.provider,
        )

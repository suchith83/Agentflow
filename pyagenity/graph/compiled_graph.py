from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Generator
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.checkpointer import BaseCheckpointer
from pyagenity.publisher import BasePublisher
from pyagenity.state import AgentState
from pyagenity.store import BaseStore
from pyagenity.utils import (
    CallbackManager,
    ResponseGranularity,
    StreamChunk,
    default_callback_manager,
)

from .invoke_handler import InvokeHandler
from .steam_handler import StreamHandler


# Import StateGraph only for typing to avoid circular import at runtime
if TYPE_CHECKING:
    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class CompiledGraph[StateT: AgentState]:
    """A compiled graph ready for execution.

    Generic over state types to support custom AgentState subclasses.
    """

    def __init__(
        self,
        state_graph: StateGraph,
        checkpointer: BaseCheckpointer | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
        publisher: BasePublisher | None = None,
    ):
        logger.info(
            "Initializing CompiledGraph with %d nodes, checkpointer=%s, store=%s",
            len(state_graph.nodes) if state_graph else 0,
            type(checkpointer).__name__ if checkpointer else None,
            type(store).__name__ if store else None,
        )
        self.state_graph = state_graph
        self.checkpointer = checkpointer
        self.store = store
        self.context_manager = state_graph.context_manager
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self.callback_manager = callback_manager or default_callback_manager
        self.publisher = publisher

        logger.debug(
            "CompiledGraph configured with interrupt_before=%s, interrupt_after=%s",
            self.interrupt_before,
            self.interrupt_after,
        )
        # create handler
        self.invoke_handler = InvokeHandler[StateT](
            state_graph=self.state_graph,
            checkpointer=self.checkpointer,
            store=self.store,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            callback_manager=self.callback_manager,
            publisher=self.publisher,
        )
        self.stream_handler = StreamHandler[StateT](
            state_graph=self.state_graph,
            checkpointer=self.checkpointer,
            store=self.store,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            callback_manager=self.callback_manager,
            publisher=self.publisher,
        )

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph synchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state.

        Args:
            input_data: Input dict
            config: Configuration dictionary

        Returns:
            Final state dict and messages
        """
        logger.info(
            "Starting synchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        logger.debug("Input data keys: %s", list(input_data.keys()) if input_data else [])
        # Async Will Handle Event Publish

        try:
            result = asyncio.run(self.ainvoke(input_data, config, response_granularity))
            logger.info("Synchronous graph execution completed successfully")
            return result
        except Exception as e:
            logger.exception("Synchronous graph execution failed: %s", e)
            raise

    async def ainvoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph asynchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state
        based on the AgentState's execution metadata.

        Args:
            input_data: Input dict with 'messages' key (for new execution) or
                       additional data for resuming
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Returns:
            Response dict based on granularity
        """

        return await self.invoke_handler.invoke(
            input_data,
            config,
            response_granularity,
        )

    def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> Generator[StreamChunk, None, None]:
        """Execute the graph synchronously with streaming support.

        Yields StreamChunk objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            StreamChunk objects with incremental content
        """

        # For sync streaming, we'll use asyncio.run to handle the async implementation
        async def _async_stream():
            async for chunk in await self.astream(input_data, config, response_granularity):
                yield chunk

        # Use a helper to convert async generator to sync generator
        gen = _async_stream()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Synchronous streaming started")

        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
        logger.info("Synchronous streaming completed")

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the graph asynchronously with streaming support.

        Yields StreamChunk objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            StreamChunk objects with incremental content
        """
        return self.stream_handler.stream(
            input_data,
            config,
            response_granularity,
        )

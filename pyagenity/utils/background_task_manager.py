"""
Background task manager for async operations in PyAgenity.

This module provides BackgroundTaskManager, which tracks and manages
asyncio background tasks, ensuring proper cleanup and error logging.
"""

import asyncio
import logging
from collections.abc import Coroutine


logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manages asyncio background tasks for agent operations.

    Tracks created tasks, ensures cleanup, and logs errors from background execution.
    """

    def __init__(self):
        """
        Initialize the BackgroundTaskManager.
        """
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro: Coroutine) -> None:
        """
        Create and track a background asyncio task.

        Args:
            coro (Coroutine): The coroutine to run in the background.
        """
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        Remove completed task and log exceptions if any.

        Args:
            task (asyncio.Task): The completed asyncio task.
        """
        self._tasks.discard(task)
        try:
            task.result()  # raises if task failed
        except Exception as e:
            logger.error(f"Background task raised an exception: {e}")

    async def wait_for_all(self) -> None:
        """
        Wait for all tracked background tasks to complete.

        Returns:
            None
        """
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} background tasks to finish...")
            await asyncio.gather(*self._tasks)
            logger.info("All background tasks finished.")

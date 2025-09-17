import asyncio
import logging
from collections.abc import Coroutine


logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro: Coroutine) -> None:
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Remove completed task and log exceptions if any."""
        self._tasks.discard(task)
        try:
            task.result()  # raises if task failed
        except Exception as e:
            logger.error(f"Background task raised an exception: {e}")

    async def wait_for_all(self) -> None:
        """Wait for all tracked tasks to complete."""
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} background tasks to finish...")
            await asyncio.gather(*self._tasks)
            logger.info("All background tasks finished.")

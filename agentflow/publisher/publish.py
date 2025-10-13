import logging

from injectq import Inject

from agentflow.publisher.base_publisher import BasePublisher
from agentflow.publisher.events import EventModel
from agentflow.utils.background_task_manager import BackgroundTaskManager


logger = logging.getLogger(__name__)


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event asynchronously if publisher is configured.

    Args:
        event: The event to publish.
        publisher: The publisher instance, or None.
    """
    if publisher:
        try:
            await publisher.publish(event)
            logger.debug("Published event: %s", event)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)


def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> None:
    """Publish an event asynchronously using the background task manager.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
    """
    # Store the task to prevent it from being garbage collected
    task_manager.create_task(_publish_event_task(event, publisher))

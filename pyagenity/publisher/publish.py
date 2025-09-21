import logging

from injectq import Inject

from pyagenity.publisher.base_publisher import BasePublisher
from pyagenity.publisher.events import EventModel
from pyagenity.utils.background_task_manager import BackgroundTaskManager


logger = logging.getLogger(__name__)


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event if publisher is configured."""
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
    """Publish an event if publisher is configured."""
    # Store the task to prevent it from being garbage collected
    task_manager.create_task(_publish_event_task(event, publisher))

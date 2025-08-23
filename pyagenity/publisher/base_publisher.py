from typing import Any


class BasePublisher:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    async def publish(self, info: dict[str, Any]) -> Any:
        pass

    def close(self):
        """
        Close the publisher and release any resources.
        This method should be overridden by subclasses to provide
        specific cleanup logic.
        And it will be called externally
        """

    def sync_close(self):
        """
        Close the publisher and release any resources.
        This method should be overridden by subclasses to provide
        specific cleanup logic.
        And it will be called externally
        """

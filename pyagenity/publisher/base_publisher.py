from typing import Any


class BasePublisher:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    async def publish(self, info: dict[str, Any]) -> Any:
        pass

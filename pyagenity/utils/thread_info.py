from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ThreadInfo(BaseModel):
    thread_id: int | str
    thread_name: str | None = None
    user_id: int | str | None = None
    metadata: dict[str, Any] | None = None
    updated_at: datetime | None = None

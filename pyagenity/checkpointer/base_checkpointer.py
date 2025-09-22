import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.thread_info import ThreadInfo


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyagenity.state import AgentState
    from pyagenity.utils import Message


StateT = TypeVar("StateT", bound="AgentState")


class BaseCheckpointer[StateT: AgentState](ABC):
    """
    Abstract base class for checkpointing agent state.

    - Async-first design: subclasses should implement `async def` methods.
    - If a subclass provides only a sync `def`, it will be executed in a worker
      thread automatically using `asyncio.to_thread`.
    - Callers always use the async APIs (`await cp.put_state(...)`, etc.).
    """

    ###########################
    #### SETUP ################
    ###########################
    def setup(self) -> Any:
        raise NotImplementedError

    async def asetup(self) -> Any:
        raise NotImplementedError

    # -------------------------
    # State methods Async
    # -------------------------
    @abstractmethod
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        raise NotImplementedError

    @abstractmethod
    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        raise NotImplementedError

    @abstractmethod
    async def aclear_state(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        raise NotImplementedError

    # -------------------------
    # State methods Sync
    # -------------------------
    @abstractmethod
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def get_state(self, config: dict[str, Any]) -> StateT | None:
        raise NotImplementedError

    @abstractmethod
    def clear_state(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        raise NotImplementedError

    # -------------------------
    # Message methods async
    # -------------------------
    @abstractmethod
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        raise NotImplementedError

    @abstractmethod
    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        raise NotImplementedError

    @abstractmethod
    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        raise NotImplementedError

    # -------------------------
    # Message methods sync
    # -------------------------
    @abstractmethod
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_message(self, config: dict[str, Any]) -> Message:
        raise NotImplementedError

    @abstractmethod
    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        raise NotImplementedError

    @abstractmethod
    def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        raise NotImplementedError

    # -------------------------
    # Thread methods async
    # -------------------------
    @abstractmethod
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        raise NotImplementedError

    @abstractmethod
    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        raise NotImplementedError

    @abstractmethod
    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        raise NotImplementedError

    # -------------------------
    # Thread methods sync
    # -------------------------
    @abstractmethod
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        raise NotImplementedError

    @abstractmethod
    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        raise NotImplementedError

    @abstractmethod
    def clean_thread(self, config: dict[str, Any]) -> Any | None:
        raise NotImplementedError

    # -------------------------
    # Clean Resources
    # -------------------------
    @abstractmethod
    def release(self) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def arelease(self) -> Any | None:
        raise NotImplementedError

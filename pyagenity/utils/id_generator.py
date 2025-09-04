import enum
import time
from abc import ABC, abstractmethod


class IDType(enum.StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BIGINT = "bigint"


class BaseIDGenerator(ABC):
    """Base class for ID generation strategies."""

    @property
    @abstractmethod
    def id_type(self) -> IDType:
        """Return the type of ID generated."""
        raise NotImplementedError("id_type method must be implemented")

    @abstractmethod
    def generate(self) -> str | int:
        """Generate a new unique ID."""
        raise NotImplementedError("generate method must be implemented")


class UUIDGenerator(BaseIDGenerator):
    """ID generator that produces UUID strings."""

    @property
    def id_type(self) -> IDType:
        return IDType.STRING

    def generate(self) -> str:
        from uuid import uuid4  # noqa: PLC0415

        return str(uuid4())


class IntegerIDGenerator(BaseIDGenerator):
    """ID generator that produces integer IDs based on current time."""

    @property
    def id_type(self) -> IDType:
        return IDType.INTEGER

    def generate(self) -> int:
        # Use current time in microseconds for uniqueness
        return int(time.time() * 1_000_000)


class BigIntIDGenerator(BaseIDGenerator):
    """ID generator that produces big integer IDs based on current time."""

    @property
    def id_type(self) -> IDType:
        return IDType.BIGINT

    def generate(self) -> int:
        # Use current time in nanoseconds for higher uniqueness
        return int(time.time() * 1_000_000_000)


class DefaultIDGenerator(BaseIDGenerator):
    """Default ID generator using UUID strings."""

    @property
    def id_type(self) -> IDType:
        return IDType.STRING

    def generate(self) -> str:
        # if you keep empty, then it will be used default
        # framework default which is UUID based
        # if framework not using then uuid 4 will be used
        return ""

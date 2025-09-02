import base64
import secrets
import string
import time
import uuid
from enum import Enum


class IDType(Enum):
    INT = "int"
    STR = "str"


class BaseIDGenerator:
    @classmethod
    def type(cls) -> IDType:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def generate(cls) -> str | int:
        raise NotImplementedError("Subclasses must implement this method")


class IntIDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.INT

    @classmethod
    def generate(cls) -> int:
        return secrets.randbits(32)


class UUIDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.STR

    @classmethod
    def generate(cls) -> str:
        return str(uuid.uuid4())


class HexIDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.STR

    @classmethod
    def generate(cls) -> str:
        return secrets.token_hex(16)


class TimestampIDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.INT

    @classmethod
    def generate(cls) -> int:
        return int(time.time() * 1000000)


class Base64IDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.STR

    @classmethod
    def generate(cls) -> str:
        return base64.b64encode(secrets.token_bytes(16)).decode("utf-8").rstrip("=")


class ShortIDGenerator(BaseIDGenerator):
    @classmethod
    def type(cls) -> IDType:
        return IDType.STR

    @classmethod
    def generate(cls) -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(8))

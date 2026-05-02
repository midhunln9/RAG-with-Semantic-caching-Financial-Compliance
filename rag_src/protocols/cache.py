from typing import Protocol


class CacheProtocol(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

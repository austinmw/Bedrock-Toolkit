""" Base class for chat history storage. """
from abc import ABC, abstractmethod
from typing import Any


class ChatHistoryStorage(ABC):
    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def save_messages(self, messages: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def clear_messages(self) -> None:
        pass

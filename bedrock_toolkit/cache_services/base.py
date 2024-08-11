""" Base class for cache services. """
from abc import ABC, abstractmethod
from typing import Optional


class CacheService(ABC):
    @abstractmethod
    def get(self, question: str) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, question: str, response: str) -> None:
        pass

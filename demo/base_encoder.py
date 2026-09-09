from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list:
        pass
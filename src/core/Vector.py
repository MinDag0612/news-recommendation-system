from abc import ABC, abstractmethod

class Vector(ABC):
    @abstractmethod
    def get_vector(self):
        pass

    @abstractmethod
    def overview(self):
        pass

    @abstractmethod
    def summary(self):
        pass
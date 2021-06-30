from abc import ABC, abstractmethod


class SchemaElement(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fully_qualified(self) -> str:
        pass

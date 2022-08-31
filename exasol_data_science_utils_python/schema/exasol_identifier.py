import unicodedata
from abc import ABC, abstractmethod

from typeguard import typechecked


class ExasolIdentifier(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def quoted_name(self) -> str:
        pass

    @abstractmethod
    def fully_qualified(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        pass

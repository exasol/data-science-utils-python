import unicodedata
from abc import ABC, abstractmethod

from typeguard import typechecked


class ExasolIdentifier(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def quoted_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def fully_qualified(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()

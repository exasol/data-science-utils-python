from abc import ABC, abstractmethod


class ExasolIdentifier(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the ExasolIdentifier
        """

    @property
    @abstractmethod
    def quoted_name(self) -> str:
        """
        The quoted name of the ExasolIdentifier, e.g. "table_name"
        """

    @property
    @abstractmethod
    def fully_qualified(self) -> str:
        """
        The full qualified name of the ExasolIdentifier, e.g. "schema_name"."table_name"
        """

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        pass

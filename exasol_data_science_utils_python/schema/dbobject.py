from abc import ABC
from typing import TypeVar, Generic

from typeguard import typechecked

from exasol_data_science_utils_python.schema.dbobject_name import DBObjectName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object

NameType = TypeVar('NameType', bound=DBObjectName)


class DBObject(Generic[NameType], ABC):

    @typechecked
    def __init__(self, name: NameType):
        self._name = name

    @property
    def name(self) -> NameType:
        return self._name

    def __eq__(self, other):
        return type(other) == type(self) and \
               self._name == other.name

    def __repr__(self):
        return generate_repr_for_object(self)

    def __hash__(self):
        return hash(self._name)
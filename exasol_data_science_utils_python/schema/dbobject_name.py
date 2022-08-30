from abc import ABC

from typeguard import typechecked

from exasol_data_science_utils_python.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class DBObjectName(ExasolIdentifier, ABC):
    @typechecked
    def __init__(self, db_object_name: str):
        super().__init__(db_object_name)

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return type(other) == type(self) and \
               self._name == other.name

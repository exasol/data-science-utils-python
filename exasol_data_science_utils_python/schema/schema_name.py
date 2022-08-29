from typeguard import typechecked

from exasol_data_science_utils_python.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class SchemaName(ExasolIdentifier):
    @typechecked
    def __init__(self, schema_name: str):
        super().__init__(schema_name)

    def fully_qualified(self) -> str:
        return self.quoted_name()

    def __eq__(self, other):
        return isinstance(other, SchemaName) and \
               self.name == other.name

    def __repr__(self):
        return generate_repr_for_object(self)

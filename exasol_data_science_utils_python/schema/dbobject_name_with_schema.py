from typeguard import typechecked

from exasol_data_science_utils_python.schema.dbobject_name import DBObjectName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class DBObjectNameWithSchema(DBObjectName):
    @typechecked
    def __init__(self, db_object_name: str, schema: SchemaName = None):
        super().__init__(db_object_name)
        self._schema_name = schema

    @property
    def schema_name(self):
        return self._schema_name

    def fully_qualified(self) -> str:
        if self.schema_name is not None:
            return f'{self._schema_name.fully_qualified()}.{self.quoted_name()}'
        else:
            return self.quoted_name()

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return type(other) == type(self) and \
               self._name == other.name and \
               self._schema_name == other.schema_name

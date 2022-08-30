from typing import cast

from exasol_data_science_utils_python.schema.dbobject_name import DBObjectName
from exasol_data_science_utils_python.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class ConnectionObjectName(DBObjectName):

    def __init__(self, name: str):
        super().__init__(name)

    def normalized_name_for_udfs(self) -> str:
        return self.name.upper()

    def fully_qualified(self) -> str:
        return self.quoted_name()

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        # Connection names are case-insensitive https://docs.exasol.com/db/latest/sql/create_connection.htm
        return type(other) == type(self) and \
               self._name.upper() == cast(ConnectionObjectName, other).name.upper()

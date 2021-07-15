from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class Column:
    @typechecked
    def __init__(self, name: ColumnName, type: ColumnType):
        self._type = type
        self._name = name

    @property
    def type(self) -> ColumnType:
        return self._type

    @property
    def name(self) -> ColumnName:
        return self._name

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, Column) and \
               self._name == other.name and \
               self._type == other.type

    def __hash__(self):
        return
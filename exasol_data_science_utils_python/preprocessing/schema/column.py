from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.column_type import ColumnType


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
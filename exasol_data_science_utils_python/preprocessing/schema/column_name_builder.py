from typing import Union

from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName


class ColumnNameBuilder:
    def __init__(self, column_name: Union[ColumnName, None] = None):
        if column_name is not None:
            self._name = column_name.name
            self._table_name = column_name.table_name
        else:
            self._name = None
            self._table_name = None

    def with_name(self, name: str) -> "ColumnNameBuilder":
        self._name = name
        return self

    def with_table_name(self, table_name: TableName) -> "ColumnNameBuilder":
        self._table_name = table_name
        return self

    def build(self) -> ColumnName:
        name = ColumnName(self._name, table_name=self._table_name)
        return name

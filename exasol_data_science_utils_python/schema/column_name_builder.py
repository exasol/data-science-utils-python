from typing import Optional

from exasol_data_science_utils_python.schema.column_name import ColumnName
from exasol_data_science_utils_python.schema.table_name import TableName


class ColumnNameBuilder:
    def __init__(self,
                 name: Optional[str] = None,
                 table_name: Optional[TableName] = None,
                 column_name: Optional[ColumnName] = None):
        """
        Creates a builder for ColumnName objects,
        either by copying a ColumnName object or
        using the newly provided values.
        """
        self._name = None
        self._table_name = None
        if column_name is not None:
            self._name = column_name.name
            self._table_name = column_name.table_name
        if name is not None:
            self._name = name
        if table_name is not None:
            self._table_name = table_name

    def with_name(self, name: str) -> "ColumnNameBuilder":
        self._name = name
        return self

    def with_table_name(self, table_name: TableName) -> "ColumnNameBuilder":
        self._table_name = table_name
        return self

    def build(self) -> ColumnName:
        name = self.create(self._name, table_name=self._table_name)
        return name

    @staticmethod
    def create(name: str, table_name: Optional[TableName] = None) -> ColumnName:
        return ColumnName(name, table_name)

from typing import Union, List

from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.table import Table
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName


class TableBuilder:
    def __init__(self, table: Union[Table, None] = None):
        if table is not None:
            self._name = table.name
            self._columns = table.columns
            self._is_view = table.is_view
        else:
            self._name = None
            self._columns = None
            self._is_view = False

    def with_name(self, name: TableName) -> "TableBuilder":
        self._name = name
        return self

    def with_is_view(self, is_view: bool) -> "TableBuilder":
        self._is_view = is_view
        return self

    def with_columns(self, columns: List[Column]) -> "TableBuilder":
        self._columns = columns
        return self

    def build(self) -> Table:
        table = Table(self._name, self._columns, self._is_view)
        return table

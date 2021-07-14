from typing import List

from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName


class Table:

    @typechecked
    def __init__(self, name: TableName, columns: List[Column], is_view: bool = False):
        self._is_view = is_view
        self._columns = columns
        self._name = name

    @property
    def is_view(self) -> bool:
        return self._is_view

    @property
    def columns(self) -> List[Column]:
        return list(self._columns)

    @property
    def name(self) -> TableName:
        return self._name

from typing import List

from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


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

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, Table) and \
               self._name == other.name and \
               self._is_view == other.is_view and \
               self._columns == other.columns

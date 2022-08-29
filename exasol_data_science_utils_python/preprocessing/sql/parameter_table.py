from typing import Any, Dict

from typeguard import typechecked

from exasol_data_science_utils_python.schema import ColumnName
from exasol_data_science_utils_python.schema import Table
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class ParameterTable:
    @typechecked
    def __init__(self,
                 source_column: ColumnName,
                 table: Table,
                 purpose: str,
                 **kwargs: Dict[str, Any]):
        self.purpose = purpose
        self.table = table
        self.source_column = source_column
        self.kwargs = kwargs
        if table.name.schema_name is None:
            raise ValueError("Table without schema is not allowed for a ParameterTable")

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, ParameterTable) and \
               self.purpose == other.purpose and \
               self.table == other.table and \
               self.source_column == other.source_column and \
               self.kwargs == other.kwargs

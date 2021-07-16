from typing import Dict, Any

from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class TransformationColumn:
    @typechecked
    def __init__(self,
                 source_column: ColumnName,
                 input_column: ColumnName,
                 column: Column,
                 purpose: str,
                 **kwargs: Dict[str, Any]):
        self.input_column = input_column
        self.purpose = purpose
        self.column = column
        self.source_column = source_column
        self.kwargs = kwargs

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, TransformationColumn) and \
               self.column == other.column and \
               self.source_column == other.source_column and \
               self.input_column == other.input_column and \
               self.purpose == other.purpose and \
               self.kwargs == other.kwargs

from typing import List

from typeguard import typechecked

from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class TransformationTable:
    @typechecked
    def __init__(self, table_name: TableName, transformation_columns: List[TransformationColumn]):
        self.transformation_columns = transformation_columns
        self.table_name = table_name

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, TransformationTable) and \
               self.transformation_columns == other.transformation_columns and \
               self.table_name == other.table_name

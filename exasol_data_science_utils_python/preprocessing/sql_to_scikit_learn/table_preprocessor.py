from typing import List

from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class TablePreprocessor:
    def __init__(self,
                 input_column_preprocessors: List[ColumnPreprocessor],
                 target_column_preprocessors: List[ColumnPreprocessor],
                 source_table: TableName, target_schema: SchemaName):
        self.target_schema = target_schema
        self.source_table = source_table
        self.target_column_preprocessors = target_column_preprocessors
        self.input_column_preprocessors = input_column_preprocessors

    def __repr__(self):
        return generate_repr_for_object(self)
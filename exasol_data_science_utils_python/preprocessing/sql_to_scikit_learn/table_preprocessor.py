from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
    ColumnSetPreprocessor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class TablePreprocessor:
    def __init__(self,
                 input_column_set_preprocessors: ColumnSetPreprocessor,
                 target_column_set_preprocessors: ColumnSetPreprocessor,
                 source_table: TableName, target_schema: SchemaName):
        self.target_schema = target_schema
        self.source_table = source_table
        self.target_column_set_preprocessors = target_column_set_preprocessors
        self.input_column_set_preprocessors = input_column_set_preprocessors

    def __repr__(self):
        return generate_repr_for_object(self)
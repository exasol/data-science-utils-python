from typing import List

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class ColumnSetPreprocessor:
    def __init__(self,
                 column_preprocessors: List[ColumnPreprocessor],
                 column_transformer:SKLearnPrefittedColumnTransformer,
                 source_table: TableName, target_schema: SchemaName):
        self.target_schema = target_schema
        self.source_table = source_table
        self.column_transformer = column_transformer
        self.column_preprocessors = column_preprocessors

    def __repr__(self):
        return generate_repr_for_object(self)
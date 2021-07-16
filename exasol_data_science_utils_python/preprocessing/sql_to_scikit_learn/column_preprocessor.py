from abc import ABC
from typing import List

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import SKLearnPrefittedTransformer
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class ColumnPreprocessor():
    def __init__(self,
                 source_column: Column,
                 target_schema: SchemaName,
                 transformer: SKLearnPrefittedTransformer):
        self.transformer = transformer
        self.target_schema = target_schema
        self.source_column = source_column

    def __repr__(self):
        return generate_repr_for_object(self)

class SQLBasedColumnPreprocessor(ColumnPreprocessor):
    def __init__(self,
                 source_column: Column,
                 target_schema: SchemaName,
                 transformer: SKLearnPrefittedTransformer,
                 parameter_tables: List[ParameterTable]):
        super().__init__(source_column, target_schema, transformer)
        self.parameter_tables = parameter_tables
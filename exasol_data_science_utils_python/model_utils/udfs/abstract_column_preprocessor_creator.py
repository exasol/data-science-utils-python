from abc import ABC, abstractmethod
from typing import List

from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class ColumnPreprocssorCreatorResult:
    def __init__(self, input_columns_transformer: ColumnTransformer,
                 target_column_transformer: ColumnTransformer):
        self.target_column_transformer = target_column_transformer
        self.input_columns_transformer = input_columns_transformer


class AbstractColumnPreprocessorCreator(ABC):

    @abstractmethod
    def generate_column_transformers(self,
                                     sql_executor: SQLExecutor,
                                     input_columns: List[Column],
                                     target_columns: List[Column],
                                     source_table: Table,
                                     target_schema: Schema) -> ColumnPreprocssorCreatorResult:
        pass

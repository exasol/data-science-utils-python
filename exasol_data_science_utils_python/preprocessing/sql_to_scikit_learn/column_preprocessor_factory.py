from abc import ABC, abstractmethod

from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import \
    ColumnPreprocessor
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class ColumnPreprocessorFactory(ABC):

    @abstractmethod
    def create(self,
               sql_executor: SQLExecutor,
               source_column: Column,
               target_schema: SchemaName,
               experiment_name: ExperimentName) -> ColumnPreprocessor:
        pass

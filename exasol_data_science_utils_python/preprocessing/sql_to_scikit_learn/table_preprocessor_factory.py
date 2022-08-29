from abc import abstractmethod, ABC

from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import \
    TablePreprocessor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class TablePreprocessorFactory(ABC):

    @abstractmethod
    def create_table_processor(self,
                               sql_executor: SQLExecutor,
                               input_table: TableName,
                               target_schema: SchemaName,
                               experiment_name: ExperimentName) -> TablePreprocessor:
        pass

from abc import abstractmethod, ABC

from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import \
    TablePreprocessor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class TablePreprocessorFactory(ABC):

    @abstractmethod
    def create_table_processor(self,
                               sql_executor: SQLExecutor,
                               input_table: TableName,
                               target_schema: SchemaName) -> TablePreprocessor:
        pass

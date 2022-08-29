from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_standard_scaler import \
    SKLearnPrefittedStandardScaler
from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_standard_scaler import SQLStandardScaler
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor, \
    SQLBasedColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class StandardScalerFactory(ColumnPreprocessorFactory):

    def create(self,
               sql_executor: SQLExecutor,
               source_column: Column,
               target_schema: SchemaName,
               experiment_name: ExperimentName) -> ColumnPreprocessor:
        parameter_tables = \
            SQLStandardScaler().fit(sql_executor, source_column.name, target_schema, experiment_name)
        avg_stddev_parameter_tables = \
            [parameter_table for parameter_table in parameter_tables
             if parameter_table.purpose == SQLStandardScaler.MEAN_AND_STDDEV_TABLE]
        avg_stddev_parameter_table = avg_stddev_parameter_tables[0]
        result_set = sql_executor.execute(
            f"""SELECT "AVG", "STDDEV"  FROM {avg_stddev_parameter_table.table.name.fully_qualified()}""")
        rows = result_set.fetchall()
        avg_value = rows[0][0]
        stddev_value = rows[0][1]
        transformer = SKLearnPrefittedStandardScaler(avg_value=avg_value, stddev_value=stddev_value)
        column_preprocessor = SQLBasedColumnPreprocessor(source_column,
                                                         target_schema,
                                                         experiment_name,
                                                         transformer,
                                                         parameter_tables)
        return column_preprocessor

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scalar import \
    SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_min_max_scaler import SQLMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor, \
    SQLBasedColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class MinMaxScalerFactory(ColumnPreprocessorFactory):

    def create(self,
               sql_executor: SQLExecutor,
               source_column: Column,
               target_schema: SchemaName,
               experiment_name: ExperimentName) -> ColumnPreprocessor:
        parameter_tables = \
            SQLMinMaxScaler().fit(sql_executor, source_column.name, target_schema, experiment_name)
        min_range_parameter_tables = \
            [parameter_table for parameter_table in parameter_tables
             if parameter_table.purpose == SQLMinMaxScaler.MIN_AND_RANGE_TABLE]
        min_range_parameter_table = min_range_parameter_tables[0]
        result_set = sql_executor.execute(
            f"""SELECT "MIN", "RANGE"  FROM {min_range_parameter_table.table.name.fully_qualified()}""")
        rows = result_set.fetchall()
        min_value = rows[0][0]
        range_value = rows[0][1]
        transformer = SKLearnPrefittedMinMaxScaler(min_value=min_value, range_value=range_value)
        column_preprocessor = SQLBasedColumnPreprocessor(source_column,
                                                         target_schema,
                                                         experiment_name,
                                                         transformer,
                                                         parameter_tables)
        return column_preprocessor

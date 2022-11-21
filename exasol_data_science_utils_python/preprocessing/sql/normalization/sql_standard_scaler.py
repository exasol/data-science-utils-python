import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table import Table
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor import SQLColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql.transform_select_clause_part import TransformSelectClausePart
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor

MEAN_AND_STDDEV_PARAMETER_TABLE_PREFIX = "STANDARD_SCALER_PARAMETERS"


class SQLStandardScaler(SQLColumnPreprocessor):
    MEAN_AND_STDDEV_TABLE = "AVG_AND_STDDEV_TABLE"

    def _get_parameter_table_name(self, target_schema: SchemaName, source_column: ColumnName,
                                  experiment_name: ExperimentName):
        table = self._get_target_table(target_schema, source_column, experiment_name,
                                       MEAN_AND_STDDEV_PARAMETER_TABLE_PREFIX)
        return table

    def _get_parameter_table_alias(self, target_schema: SchemaName, source_column: ColumnName):
        alias = self._get_table_alias(target_schema, source_column,
                                      MEAN_AND_STDDEV_PARAMETER_TABLE_PREFIX)
        return alias

    def _get_avg_column(self, table: TableName = None):
        avg_column = ColumnNameBuilder.create("AVG", table)
        return avg_column

    def _get_stddev_column(self, table: TableName = None):
        stddev_column = ColumnNameBuilder.create("STDDEV", table)
        return stddev_column

    def requires_global_transformation_for_training_data(self) -> bool:
        return False

    def fit(self, sqlexecutor: SQLExecutor, source_column: ColumnName, target_schema: SchemaName,
            experiment_name: ExperimentName) -> List[ParameterTable]:
        parameter_table_name = self._get_parameter_table_name(target_schema, source_column, experiment_name)
        avg_column = self._get_avg_column()
        stddev_column = self._get_stddev_column()
        query = textwrap.dedent(f"""
    CREATE OR REPLACE TABLE {parameter_table_name.fully_qualified} AS
    SELECT
        CAST(AVG({source_column.fully_qualified}) as DOUBLE) as {avg_column.fully_qualified},
        CAST(STDDEV_POP({source_column.fully_qualified}) as DOUBLE) as {stddev_column.fully_qualified}
    FROM {source_column.table_like_name.fully_qualified}
    """)
        sqlexecutor.execute(query)
        avg_column = ColumnNameBuilder(column_name=avg_column).with_table_like_name(parameter_table_name).build()
        stddev_column = ColumnNameBuilder(column_name=stddev_column).with_table_like_name(parameter_table_name).build()
        parameter_table = ParameterTable(
            source_column=source_column,
            purpose=self.MEAN_AND_STDDEV_TABLE,
            table=Table(
                parameter_table_name,
                columns=[
                    Column(avg_column, ColumnType("DOUBLE")),
                    Column(stddev_column, ColumnType("DOUBLE"))
                ]
            )
        )
        return [parameter_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName,
                                          experiment_name: ExperimentName) -> List[str]:
        parameter_table = self._get_parameter_table_name(target_schema, source_column, experiment_name)
        alias = self._get_parameter_table_alias(target_schema, source_column)
        from_caluse_part = textwrap.dedent(
            f'''CROSS JOIN {parameter_table.fully_qualified} AS {alias.fully_qualified}''')
        return [from_caluse_part]

    def create_transform_select_clause_part(self,
                                            sql_executor: SQLExecutor,
                                            source_column: ColumnName,
                                            input_table: TableName,
                                            target_schema: SchemaName,
                                            experiment_name: ExperimentName) -> List[TransformSelectClausePart]:
        alias = self._get_parameter_table_alias(target_schema, source_column)
        input_column = ColumnNameBuilder.create(source_column.name, input_table)
        avg_column = self._get_avg_column(alias)
        stddev_column = self._get_stddev_column(alias)
        target_column_name = ColumnNameBuilder.create(f"{source_column.name}_STANDARD_SCALED")
        select_clause_part_str = textwrap.dedent(
            f'''({input_column.fully_qualified}-{avg_column.fully_qualified})/
        (CASE 
        WHEN {stddev_column.fully_qualified} = 0 THEN 1
        ELSE {stddev_column.fully_qualified}
        END)
        AS {target_column_name.quoted_name}''')
        select_clause_part = TransformSelectClausePart(
            tranformation_column=TransformationColumn(
                column=Column(target_column_name, ColumnType("DOUBLE")),
                input_column=input_column,
                source_column=source_column,
                purpose="StandardScaled"
            ),
            select_clause_part_expression=select_clause_part_str
        )
        return [select_clause_part]

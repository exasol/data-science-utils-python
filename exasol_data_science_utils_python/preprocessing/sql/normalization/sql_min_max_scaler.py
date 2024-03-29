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

MIN_MAX_SCALER_PARAMETER_TABLE_PREFIX = "MIN_MAX_SCALER_PARAMETERS"


class SQLMinMaxScaler(SQLColumnPreprocessor):
    """
    This ColumnPreprocessor implements a MinMaxScaler.
    It was inspired by the
    `MinMaxScaler of scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_
    """

    MIN_AND_RANGE_TABLE = "MIN_AND_RANGE_TABLE"

    def _get_parameter_table_name(self,
                                  target_schema: SchemaName,
                                  source_column: ColumnName,
                                  experiment_name: ExperimentName):
        table = self._get_target_table(target_schema,
                                       source_column,
                                       experiment_name,
                                       MIN_MAX_SCALER_PARAMETER_TABLE_PREFIX)
        return table

    def _get_parameter_table_alias(self, target_schema: SchemaName, source_column: ColumnName):
        alias = self._get_table_alias(target_schema, source_column, MIN_MAX_SCALER_PARAMETER_TABLE_PREFIX)
        return alias

    def _get_min_column(self, table: TableName = None):
        min_column = ColumnNameBuilder.create("MIN", table)
        return min_column

    def _get_range_column(self, table: TableName = None):
        range_column = ColumnNameBuilder.create("RANGE", table)
        return range_column

    def requires_global_transformation_for_training_data(self) -> bool:
        return False

    def fit(self,
            sqlexecutor: SQLExecutor,
            source_column: ColumnName,
            target_schema: SchemaName,
            experiment_name: ExperimentName) -> List[
        ParameterTable]:
        """
        This method creates a query which computes the parameter minimum and the range of the source column
        and stores them in a parameter table.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of created tables or views
        """
        parameter_table_name = self._get_parameter_table_name(target_schema, source_column, experiment_name)
        min_column = self._get_min_column()
        range_column = self._get_range_column()
        query = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE {parameter_table_name.fully_qualified} AS
            SELECT
                CAST(MIN({source_column.fully_qualified}) as DOUBLE) as {min_column.fully_qualified},
                CAST(MAX({source_column.fully_qualified})-MIN({source_column.fully_qualified}) as DOUBLE) as {range_column.fully_qualified}
            FROM {source_column.table_like_name.fully_qualified}
            """)
        sqlexecutor.execute(query)
        min_column = ColumnNameBuilder(column_name=min_column).with_table_like_name(parameter_table_name).build()
        range_column = ColumnNameBuilder(column_name=range_column).with_table_like_name(parameter_table_name).build()
        parameter_table = ParameterTable(
            source_column=source_column,
            purpose=self.MIN_AND_RANGE_TABLE,
            table=Table(
                parameter_table_name,
                columns=[
                    Column(min_column, ColumnType("DOUBLE")),
                    Column(range_column, ColumnType("DOUBLE"))
                ]
            )
        )
        return [parameter_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName,
                                          experiment_name: ExperimentName) -> \
            List[str]:
        """
        This method generates a CROSS JOIN with the parameter table which contain MIN and RANGE of the source_table.
        The CROSS JOIN is cheap, because the parameter table only contains one row.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of from-clause parts which can be concatenated with "\n"
        """
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
        """
        This method generates the normalization for the select clause which uses the paramter from the parameter table.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of select-clause parts which can be concatenated with ","
        """
        alias = self._get_parameter_table_alias(target_schema, source_column)
        input_column = ColumnNameBuilder.create(source_column.name, input_table)
        min_column = self._get_min_column(alias)
        range_column = self._get_range_column(alias)
        target_column_name = ColumnNameBuilder.create(f"{source_column.name}_MIN_MAX_SCALED")
        select_clause_part_str = textwrap.dedent(
            f'''({input_column.fully_qualified}-{min_column.fully_qualified})/{range_column.fully_qualified} AS {target_column_name.quoted_name}''')
        select_clause_part = TransformSelectClausePart(
            tranformation_column=TransformationColumn(
                column=Column(target_column_name, ColumnType("DOUBLE")),
                input_column=input_column,
                source_column=source_column,
                purpose="MinMaxScaled"
            ),
            select_clause_part_expression=select_clause_part_str
        )
        return [select_clause_part]

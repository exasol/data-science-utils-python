import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor

MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX = "MIN_MAX_SCALAR_PARAMETERS"


class MinMaxScaler(ColumnPreprocessor):
    """
    This ColumnPreprocessor implements a MinMaxScaler.
    It was inspired by the
    `MinMaxScaler of scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_
    """

    def _get_parameter_table_name(self, target_schema: SchemaName, source_column: ColumnName):
        table = self._get_target_table(target_schema, source_column, MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)
        return table

    def _get_parameter_table_alias(self, target_schema: SchemaName, source_column: ColumnName):
        alias = self._get_table_alias(target_schema, source_column, MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)
        return alias

    def _get_min_column(self, table: TableName = None):
        min_column = ColumnName("MIN", table)
        return min_column

    def _get_range_column(self, table: TableName = None):
        range_column = ColumnName("RANGE", table)
        return range_column

    def fit(self, sqlexecutor: SQLExecutor, source_column: ColumnName, target_schema: SchemaName) -> List[TableName]:
        """
        This method creates a query which computes the parameter minimum and the range of the source column
        and stores them in a parameter table.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of created tables or views
        """
        parameter_table = self._get_parameter_table_name(target_schema, source_column)
        min_column = self._get_min_column()
        range_column = self._get_range_column()
        query = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE {parameter_table.fully_qualified()} AS
            SELECT
                MIN({source_column.fully_qualified()}) as {min_column.fully_qualified()},
                MAX({source_column.fully_qualified()})-MIN({source_column.fully_qualified()}) as {range_column.fully_qualified()}
            FROM {source_column.table.fully_qualified()}
            """)
        sqlexecutor.execute(query)
        return [parameter_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName) -> \
            List[str]:
        """
        This method generates a CROSS JOIN with the parameter table which contain MIN and RANGE of the source_table.
        The CROSS JOIN is cheap, because the parameter table only contains one row.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of from-clause parts which can be concatenated with "\n"
        """
        parameter_table = self._get_parameter_table_name(target_schema, source_column)
        alias = self._get_parameter_table_alias(target_schema, source_column)
        from_caluse_part = textwrap.dedent(
            f'''CROSS JOIN {parameter_table.fully_qualified()} AS {alias.fully_qualified()}''')
        return [from_caluse_part]

    def create_transform_select_clause_part(self,
                                            sql_executor: SQLExecutor,
                                            source_column: ColumnName,
                                            input_table: TableName,
                                            target_schema: SchemaName) -> List[str]:
        """
        This method generates the normalization for the select clause which uses the paramter from the parameter table.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of select-clause parts which can be concatenated with ","
        """
        alias = self._get_parameter_table_alias(target_schema, source_column)
        input_column = ColumnName(source_column.name, input_table)
        min_column = self._get_min_column(alias)
        range_column = self._get_range_column(alias)
        select_clause_part = textwrap.dedent(
            f'''({input_column.fully_qualified()}-{min_column.fully_qualified()})/{range_column.fully_qualified()} AS "{source_column.name}_MIN_MAX_SCALED"''')
        return [select_clause_part]

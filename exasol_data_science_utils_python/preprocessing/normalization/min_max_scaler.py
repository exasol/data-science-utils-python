import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table

MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX = "MIN_MAX_SCALAR_PARAMETERS"


class MinMaxScaler(ColumnPreprocessor):

    def _get_parameter_table_name(self, target_schema: Schema, source_column: Column):
        table = self._get_target_table(target_schema, source_column, MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)
        return table

    def _get_parameter_table_alias(self, target_schema: Schema, source_column: Column):
        alias = self._get_table_alias(target_schema, source_column, MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)
        return alias

    def _get_min_column(self, table: Table = None):
        min_column = Column("MIN", table)
        return min_column

    def _get_range_column(self, table: Table = None):
        range_column = Column("RANGE", table)
        return range_column

    def create_fit_queries(self, source_column: Column, target_schema: Schema) -> List[str]:
        parameter_table = self._get_parameter_table_name(target_schema, source_column)
        min_column = self._get_min_column()
        range_column = self._get_range_column()
        query = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE {parameter_table.identifier()} AS
            SELECT
                MIN({source_column.identifier()}) as {min_column.identifier()},
                MAX({source_column.identifier()})-MIN({source_column.identifier()}) as {range_column.identifier()}
            FROM {source_column.table.identifier()}
            """)

        return [query]

    def create_transform_from_clause_part(self, source_column: Column, input_table: Table, target_schema: Schema) -> \
    List[str]:
        parameter_table = self._get_parameter_table_name(target_schema, source_column)
        alias = self._get_parameter_table_alias(target_schema, source_column)
        from_caluse_part = textwrap.dedent(
            f'''CROSS JOIN {parameter_table.identifier()} AS {alias.identifier()}''')
        return [from_caluse_part]

    def create_transform_select_clause_part(self, source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        alias = self._get_parameter_table_alias(target_schema, source_column)
        input_column = Column(source_column.name, input_table)
        min_column = self._get_min_column(alias)
        range_column = self._get_range_column(alias)
        select_clause_part = textwrap.dedent(
            f'''({input_column.identifier()}-{min_column.identifier()})/{range_column.identifier()} AS "{source_column.name}_MIN_MAX_SCALED"''')
        return [select_clause_part]

import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table

ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX = "ORDINAL_ENCODER_DICTIONARY"


class OrdinalEncoder(ColumnPreprocessor):

    def _get_dictionary_table_alias(self, target_schema: Schema, source_column: Column):
        return self._get_table_alias(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_dictionary_table_name(self, target_schema: Schema, source_column: Column):
        return self._get_target_table(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_id_column(self, table: Table = None):
        min_column = Column("ID", table)
        return min_column

    def _get_value_column(self, table: Table = None):
        range_column = Column("VALUE", table)
        return range_column

    def create_fit_queries(self, source_column: Column, target_schema: Schema) -> List[
        str]:
        dictionary_table = self._get_dictionary_table_name(target_schema, source_column)
        id_column = self._get_id_column()
        value_column = self._get_value_column()
        value_column_alias = Column("VALUE")
        query = textwrap.dedent(f"""
                CREATE OR REPLACE TABLE {dictionary_table.identifier()} AS
                SELECT
                    rownum - 1 as {id_column.identifier()},
                    {value_column_alias.identifier()}
                FROM (
                    SELECT distinct {source_column.identifier()} as {value_column_alias.identifier()}
                    FROM {source_column.table.identifier()}
                    ORDER BY {source_column.identifier()}
                );
                """)
        return [query]

    def create_transform_from_clause_part(self,
                                          source_column: Column,
                                          input_table: Table,
                                          target_schema: Schema) -> List[str]:
        dictionary_table = self._get_dictionary_table_name(target_schema, source_column)
        alias = self._get_dictionary_table_alias(target_schema, source_column)
        input_column = Column(source_column.name, input_table)
        id_column = self._get_id_column(alias)
        value_column = self._get_value_column(alias)
        from_clause_part = textwrap.dedent(f"""
            JOIN {dictionary_table.identifier()}
            AS {alias.identifier()}
            ON
                {value_column.identifier()} = 
                {input_column.identifier()}
            """)
        return [from_clause_part]

    def create_transform_select_clause_part(self,
                                            source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        alias = self._get_dictionary_table_alias(target_schema, source_column)
        id_column = self._get_id_column(alias)
        select_clause_part = textwrap.dedent(f'{id_column.identifier()} AS "{source_column.name}_ID"')
        return [select_clause_part]

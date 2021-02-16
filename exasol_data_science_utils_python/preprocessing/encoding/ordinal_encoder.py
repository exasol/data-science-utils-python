from typing import List
import textwrap

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor

ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX = "ORDINAL_ENCODER_DICTIONARY"


class OrdinalEncoder(ColumnPreprocessor):

    def _get_dictionary_table_alias(self, target_schema: str, source_schema: str, source_table: str, source_column: str):
        return self._get_table_alias(
            target_schema, source_schema, source_table, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)


    def _get_dictionary_table_name(self, target_schema: str, source_schema: str, source_table: str, source_column: str):
        return self._get_target_table_name(
            target_schema, source_schema, source_table, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def create_fit_queries(self, source_schema: str, source_table: str, source_column: str, target_schema: str) -> List[
        str]:
        dictionary_table = self._get_dictionary_table_name(target_schema, source_schema, source_table, source_column)
        source_table_qualified = self._get_table_qualified(source_schema, source_table)
        query = textwrap.dedent(f"""
                CREATE OR REPLACE TABLE {dictionary_table} AS
                SELECT
                    rownum - 1 as "ID",
                    "{source_column}" as "VALUE"
                FROM (
                    SELECT distinct "{source_column}"
                    FROM {source_table_qualified}
                    ORDER BY "{source_column}"
                );
                """)
        return [query]

    def create_from_clause_part(self, source_schema: str, source_table: str, source_column: str,
                                input_schema: str, input_table: str,
                                target_schema: str) -> List[str]:
        dictionary_table = self._get_dictionary_table_name(target_schema, source_schema, source_table, source_column)
        input_table_qualified = self._get_table_qualified(input_schema, input_table)
        alias = self._get_dictionary_table_alias(target_schema, source_schema, source_table, source_column)
        from_clause_part = textwrap.dedent(f"""
            JOIN {dictionary_table}
            AS {alias}
            ON
                {alias}."VALUE" = 
                {input_table_qualified}."{source_column}"
            """)
        return [from_clause_part]

    def create_select_clause_part(self, source_schema: str, source_table: str, source_column: str,
                                  input_schema: str, input_table: str,
                                  target_schema: str) -> List[str]:
        alias = self._get_dictionary_table_alias(target_schema, source_schema, source_table, source_column)
        select_clause_part = textwrap.dedent(f'{alias}."ID" AS "{source_column}_ID"')
        return [select_clause_part]

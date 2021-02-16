from typing import List
import textwrap

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor

MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX = "MIN_MAX_SCALAR_PARAMETERS"

class MinMaxScaler(ColumnPreprocessor):

    def _get_parameter_table_name(self, target_schema:str, source_schema:str, source_table:str, source_column:str):
        return self._get_target_table_name(
            target_schema, source_schema, source_table, source_column,
            MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)

    def _get_parameter_table_alias(self, target_schema: str, source_schema: str, source_table: str, source_column: str):
        return self._get_table_alias(
            target_schema, source_schema, source_table, source_column,
            MIN_MAX_SCALAR_PARAMETER_TABLE_PREFIX)

    def create_fit_queries(self, source_schema:str, source_table:str, source_column:str, target_schema:str)->List[str]:
        parameter_table = self._get_parameter_table_name(target_schema, source_schema, source_table, source_column)
        source_table_qualified= self._get_table_qualified(source_schema, source_table)
        query = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE {parameter_table} AS
            SELECT
                MIN({source_table_qualified}."{source_column}") as "MIN",
                (MAX({source_table_qualified}."{source_column}")-MIN({source_table_qualified}."{source_column}")) as "RANGE"
            FROM {source_table_qualified}
            """)
        return [query]

    def create_from_clause_part(self, source_schema:str, source_table:str, source_column:str,
                                input_schema: str, input_table: str,
                                target_schema:str)->List[str]:
        parameter_table = self._get_parameter_table_name(target_schema, source_schema, source_table, source_column)
        alias = self._get_parameter_table_alias(target_schema, source_schema, source_table, source_column)
        return [textwrap.dedent(f'''
        CROSS JOIN {parameter_table} 
        AS {alias}
        ''')]

    def create_select_clause_part(self, source_schema:str, source_table:str, source_column:str,
                                  input_schema: str, input_table: str,
                                  target_schema:str)->List[str]:
        input_table_qualified= self._get_table_qualified(input_schema, input_table)
        alias = self._get_parameter_table_alias(target_schema, source_schema, source_table, source_column)
        create_select_clause_part = textwrap.dedent(f'''
                (
                    ({input_table_qualified}."{source_column}" -
                        {alias}."MIN") /
                    {alias}."RANGE"
                ) AS "{source_column}_MIN_MAX_SCALED"''')
        return [create_select_clause_part]
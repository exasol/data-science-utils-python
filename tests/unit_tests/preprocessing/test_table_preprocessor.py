import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.table_preprocessor import TablePreprocessor, \
    ColumnPreprocesserDefinition

class MyColumnPreprocessor(ColumnPreprocessor):
    def create_fit_queries(self, source_schema:str, source_table:str, source_column:str, target_schema:str) ->List[str]:
        target_table = self._get_target_table_name(target_schema, source_schema, source_table, source_column, "PREFIX")
        return [f'''CREATE OR REPLACE TABLE {target_table} AS SELECT 1 AS "VALUE"''']

    def create_from_clause_part(self, source_schema:str, source_table:str, source_column:str, target_schema:str) ->List[str]:
        target_table = self._get_target_table_name(target_schema, source_schema, source_table, source_column, "PREFIX")
        return [f"CROSS JOIN {target_table}"]

    def create_select_clause_part(self, source_schema:str, source_table:str, source_column:str, target_schema:str) ->List[str]:
        return [f'1 as "{source_column}_VALUE"']

def test_table_preprocessor():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column1 = "SOURCE_COLUMN1"
    source_column2 = "SOURCE_COLUMN2"
    column_preprocessor_defintions = [ColumnPreprocesserDefinition(source_column1, MyColumnPreprocessor()),
                                      ColumnPreprocesserDefinition(source_column2, MyColumnPreprocessor()),
                                      ]

    table_preprocessor = TablePreprocessor(target_schema, source_schema, source_table, column_preprocessor_defintions)
    queries = table_preprocessor.create_fit_queries()
    source_column1_create_table = textwrap.dedent('''CREATE OR REPLACE TABLE "SOURCE_SCHEMA"."SOURCE_TABLE" AS SELECT 1 AS "VALUE"''')
    assert source_column1_create_table in queries

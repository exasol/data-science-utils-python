import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.table_preprocessor import TablePreprocessor, \
    ColumnPreprocesserDefinition


class MyColumnPreprocessor(ColumnPreprocessor):
    def create_fit_queries(self, source_schema: str, source_table: str, source_column: str, target_schema: str) -> List[
        str]:
        target_table = self._get_target_table_name(target_schema, source_schema, source_table, source_column, "PREFIX")
        return [textwrap.dedent(f'''
            CREATE OR REPLACE TABLE {target_table} AS 
            SELECT 1 AS "VALUE"
            ''')]

    def create_from_clause_part(self, source_schema: str, source_table: str, source_column: str,
                                input_schema: str, input_table: str,
                                target_schema: str) -> \
            List[str]:
        target_table = self._get_target_table_name(target_schema, source_schema, source_table, source_column, "PREFIX")
        return [f"CROSS JOIN {target_table}"]

    def create_select_clause_part(self, source_schema: str, source_table: str, source_column: str,
                                  input_schema: str, input_table: str,
                                  target_schema: str) -> List[str]:
        return [f'1 AS "{source_column}_VALUE"']


def test_table_preprocessor_create_fit_queries():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column1 = "SOURCE_COLUMN1"
    source_column2 = "SOURCE_COLUMN2"
    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1, MyColumnPreprocessor()),
        ColumnPreprocesserDefinition(source_column2, MyColumnPreprocessor()),
    ]

    table_preprocessor = TablePreprocessor(target_schema, source_schema, source_table, column_preprocessor_defintions)
    queries = table_preprocessor.create_fit_queries()
    source_column1_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')
    source_column2_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN2_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')

    assert source_column1_create_table == queries[0] and \
           source_column2_create_table == queries[1]


def test_table_preprocessor_create_transform_query():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column1 = "SOURCE_COLUMN1"
    source_column2 = "SOURCE_COLUMN2"
    input_schema = "INPUT_SCHEMA"
    input_table = "INPUT_TABLE"

    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1, MyColumnPreprocessor()),
        ColumnPreprocesserDefinition(source_column2, MyColumnPreprocessor()),
    ]

    table_preprocessor = TablePreprocessor(target_schema, source_schema, source_table, column_preprocessor_defintions)
    query = table_preprocessor.create_transform_query(input_schema, input_table)
    expected = textwrap.dedent('''
CREATE OR REPLACE TABLE "TARGET_SCHEMA"."INPUT_SCHEMA_INPUT_TABLE_TRANSFORMED" AS
SELECT
1 AS "SOURCE_COLUMN1_VALUE",
1 AS "SOURCE_COLUMN2_VALUE"
FROM "INPUT_SCHEMA"."INPUT_TABLE"
CROSS JOIN "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_PREFIX"
CROSS JOIN "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN2_PREFIX"
''')
    assert expected == query

import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column_name import Column
from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema
from exasol_data_science_utils_python.preprocessing.schema.table_name import Table
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor
from exasol_data_science_utils_python.preprocessing.table_preprocessor import TablePreprocessor, \
    ColumnPreprocesserDefinition
from tests.unit_tests.preprocessing.mock_sql_executor import MockSQLExecutor


class MyColumnPreprocessor(ColumnPreprocessor):
    def fit(self, sqlexecutor: SQLExecutor, source_column: Column, target_schema: Schema) -> List[
        str]:
        target_table = self._get_target_table(target_schema, source_column, "PREFIX")
        query = textwrap.dedent(f'''
                   CREATE OR REPLACE TABLE {target_table.fully_qualified()} AS 
                   SELECT 1 AS "VALUE"
                   ''')
        sqlexecutor.execute(query)
        return [target_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: Column,
                                          input_table: Table,
                                          target_schema: Schema) -> List[str]:
        target_table = self._get_target_table(target_schema, source_column, "PREFIX")
        return [f"CROSS JOIN {target_table.fully_qualified()}"]

    def create_transform_select_clause_part(self,
                                            sql_executor: SQLExecutor,
                                            source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        return [f'1 AS "{source_column.name}_VALUE"']


def test_table_preprocessor_create_fit_queries():
    source_schema = Schema("SRC_SCHEMA")
    source_table = Table("SRC_TABLE", source_schema)
    target_schema = Schema("TGT_SCHEMA")
    source_column1 = Column("SRC_COLUMN1", source_table)
    source_column2 = Column("SRC_COLUMN2", source_table)
    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1.name, MyColumnPreprocessor()),
        ColumnPreprocesserDefinition(source_column2.name, MyColumnPreprocessor()),
    ]

    table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
    mock_sql_executor = MockSQLExecutor()
    tables = table_preprocessor.fit(mock_sql_executor)
    source_column1_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')
    source_column2_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN2_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')
    # TODO assert tables
    assert source_column1_create_table == mock_sql_executor.queries[0] and \
           source_column2_create_table == mock_sql_executor.queries[1]


def test_table_preprocessor_create_transform_query():
    source_schema = Schema("SRC_SCHEMA")
    source_table = Table("SRC_TABLE", source_schema)
    target_schema = Schema("TGT_SCHEMA")
    source_column1 = Column("SRC_COLUMN1", source_table)
    source_column2 = Column("SRC_COLUMN2", source_table)
    input_schema = Schema("IN_SCHEMA")
    input_table = Table("IN_TABLE", input_schema)

    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1.name, MyColumnPreprocessor()),
        ColumnPreprocesserDefinition(source_column2.name, MyColumnPreprocessor()),
    ]

    table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
    mock_sql_executor = MockSQLExecutor()
    table = table_preprocessor.transform(mock_sql_executor, input_table)
    expected = textwrap.dedent(
'''CREATE OR REPLACE VIEW "TGT_SCHEMA"."IN_SCHEMA_IN_TABLE_TRANSFORMED" AS
SELECT
1 AS "SRC_COLUMN1_VALUE",
1 AS "SRC_COLUMN2_VALUE"
FROM "IN_SCHEMA"."IN_TABLE"
CROSS JOIN "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_PREFIX"
CROSS JOIN "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN2_PREFIX"''')
    # TODO assert table
    assert expected == mock_sql_executor.queries[0]

import textwrap

from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName
from tests.unit_tests.preprocessing.mock_sql_executor import MockSQLExecutor


def test_ordinal_encoder_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    encoder = OrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    tables = encoder.fit(mock_sql_executor, source_column, target_schema)
    expected = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY" AS
            SELECT
                CAST(rownum - 1 AS INTEGER) as "ID",
                "VALUE"
            FROM (
                SELECT DISTINCT "SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1" as "VALUE"
                FROM "SRC_SCHEMA"."SRC_TABLE"
                ORDER BY "SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1"
            );
            """)
    assert mock_sql_executor.queries == [expected]


def test_ordinal_encoder_create_from_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    encoder = OrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    from_clause_part = encoder.create_transform_from_clause_part(
        mock_sql_executor, source_column, input_table, target_schema)
    expected = textwrap.dedent("""
            LEFT OUTER JOIN "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            ON
                "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."VALUE" = 
                "IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"
            """)
    assert from_clause_part == [expected]


def test_ordinal_encoder_create_select_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    encoder = OrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    select_clause_part = encoder.create_transform_select_clause_part(
        mock_sql_executor, source_column, input_table, target_schema)
    expected = textwrap.dedent(
        '"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."ID" AS "SRC_COLUMN1_ID"')
    assert select_clause_part == [expected]

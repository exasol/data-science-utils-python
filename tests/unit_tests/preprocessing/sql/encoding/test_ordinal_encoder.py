import textwrap

from exasol_data_science_utils_python.preprocessing.sql.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table import Table
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from tests.unit_tests.preprocessing.mock_sql_executor import MockSQLExecutor


def test_ordinal_encoder_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    encoder = OrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = encoder.fit(mock_sql_executor, source_column, target_schema)
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
    expected_parameter_table = get_expected_parameter_Table()
    assert mock_sql_executor.queries == [expected]
    assert parameter_tables == [expected_parameter_table]


def get_expected_parameter_Table():
    expected_parameter_table = ParameterTable(
        source_column=ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        table=Table(
            name=TableName("SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY", SchemaName("TGT_SCHEMA")),
            columns=[
                Column(name=ColumnName("ID"), type=ColumnType("INTEGER")),
                Column(name=ColumnName("VALUE"), type=ColumnType("ANY")),
            ]
        ),
        purpose="DictionaryTable"
    )
    return expected_parameter_table


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
    expected_tranformation_column = TransformationColumn(
        source_column=ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        input_column=ColumnName("SRC_COLUMN1", TableName("IN_TABLE", SchemaName("IN_SCHEMA"))),
        column=Column(ColumnName("SRC_COLUMN1_ID"),ColumnType("INTEGER")),
        purpose="ReplaceValueByID"
    )
    assert len(select_clause_part)==1
    assert select_clause_part[0].select_clause_part_expression == expected
    assert select_clause_part[0].tranformation_column == expected_tranformation_column

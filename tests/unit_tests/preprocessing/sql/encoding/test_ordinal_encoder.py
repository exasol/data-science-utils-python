import textwrap

from exasol_data_science_utils_python.preprocessing.sql.encoding.sql_ordinal_encoder import SQLOrdinalEncoder
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table import Table
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.udf_utils.testing.mock_sql_executor import MockSQLExecutor


def test_ordinal_encoder_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    experiment_name = ExperimentName("EXPERIMENT")
    encoder = SQLOrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = encoder.fit(mock_sql_executor, source_column, target_schema, experiment_name)
    expected = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY" AS
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
        source_column=ColumnNameBuilder.create("SRC_COLUMN1", TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        table=Table(
            name=TableNameBuilder.create("EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY", SchemaName("TGT_SCHEMA")),
            columns=[
                Column(name=ColumnNameBuilder.create("ID"), type=ColumnType("INTEGER")),
                Column(name=ColumnNameBuilder.create("VALUE"), type=ColumnType("ANY")),
            ]
        ),
        purpose="DictionaryTable"
    )
    return expected_parameter_table


def test_ordinal_encoder_create_from_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableNameBuilder.create("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    encoder = SQLOrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    from_clause_part = encoder.create_transform_from_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    expected = textwrap.dedent("""
            LEFT OUTER JOIN "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            ON
                "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."VALUE" = 
                "IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"
            """)
    assert from_clause_part == [expected]


def test_ordinal_encoder_create_select_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableNameBuilder.create("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    encoder = SQLOrdinalEncoder()
    mock_sql_executor = MockSQLExecutor()
    select_clause_part = encoder.create_transform_select_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    expected = textwrap.dedent(
        '"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."ID" AS "SRC_COLUMN1_ID"')
    expected_tranformation_column = TransformationColumn(
        source_column=ColumnNameBuilder.create("SRC_COLUMN1", TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        input_column=ColumnNameBuilder.create("SRC_COLUMN1", TableNameBuilder.create("IN_TABLE", SchemaName("IN_SCHEMA"))),
        column=Column(ColumnNameBuilder.create("SRC_COLUMN1_ID"),ColumnType("INTEGER")),
        purpose="ReplaceValueByID"
    )
    assert len(select_clause_part)==1
    assert select_clause_part[0].select_clause_part_expression == expected
    assert select_clause_part[0].tranformation_column == expected_tranformation_column

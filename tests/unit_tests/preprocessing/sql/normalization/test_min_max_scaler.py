import textwrap

from exasol_data_science_utils_python.preprocessing.sql.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table import Table
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from tests.unit_tests.mock_sql_executor import MockSQLExecutor


def test_min_max_scaler_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    scaler = MinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = scaler.fit(mock_sql_executor, source_column, target_schema)

    expected_parameter_tables = get_expected_parameter_tables()

    expected = textwrap.dedent("""
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" AS
        SELECT
            CAST(MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "MIN",
            CAST(MAX("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1")-MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "RANGE"
        FROM "SRC_SCHEMA"."SRC_TABLE"
        """)
    assert mock_sql_executor.queries == [expected]
    assert expected_parameter_tables == parameter_tables


def get_expected_parameter_tables():
    target_table_name = TableName("SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS",
                                  SchemaName("TGT_SCHEMA"))
    expected_parameter_table = ParameterTable(
        source_column=ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        table=Table(target_table_name,
                    columns=
                    [
                        Column(ColumnName("MIN", target_table_name), ColumnType("DOUBLE")),
                        Column(ColumnName("RANGE", target_table_name), ColumnType("DOUBLE"))
                    ]),
        purpose="StoreMinAndRange"
    )
    return [expected_parameter_table]


def test_min_max_scaler_create_from_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    scaler = MinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    from_clause_part = scaler.create_transform_from_clause_part(
        mock_sql_executor, source_column, input_table, target_schema)
    assert from_clause_part[0] == textwrap.dedent(
        f'''CROSS JOIN "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"''')


def test_min_max_scaler_create_select_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    scaler = MinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    select_clause_parts = scaler.create_transform_select_clause_part(
        mock_sql_executor, source_column, input_table, target_schema)
    expected = textwrap.dedent(
        '''("IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"-"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."MIN")/"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."RANGE" AS "SRC_COLUMN1_MIN_MAX_SCALED"''')
    expected_transformation_column = \
        TransformationColumn(
            source_column=ColumnName("SRC_COLUMN1", source_table),
            input_column=ColumnName("SRC_COLUMN1", input_table),
            purpose="MinMaxScaled",
            column=Column(ColumnName("SRC_COLUMN1_MIN_MAX_SCALED"),ColumnType("DOUBLE"))
        )
    assert len(select_clause_parts) == 1
    assert select_clause_parts[0].select_clause_part_expression == expected
    assert select_clause_parts[0].tranformation_column == expected_transformation_column

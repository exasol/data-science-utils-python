import textwrap

from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_min_max_scaler import SQLMinMaxScaler
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


def test_min_max_scaler_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLMinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = scaler.fit(mock_sql_executor, source_column, target_schema, experiment_name)

    expected_parameter_tables = get_expected_parameter_tables()

    expected = textwrap.dedent("""
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS" AS
        SELECT
            CAST(MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "MIN",
            CAST(MAX("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1")-MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "RANGE"
        FROM "SRC_SCHEMA"."SRC_TABLE"
        """)
    assert mock_sql_executor.queries == [expected]
    assert expected_parameter_tables == parameter_tables


def get_expected_parameter_tables():
    target_table_name = TableNameBuilder.create(
        "EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS",
        SchemaName("TGT_SCHEMA"))
    expected_parameter_table = ParameterTable(
        source_column=ColumnNameBuilder.create("SRC_COLUMN1", TableNameBuilder.create(
            "SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        table=Table(target_table_name,
                    columns=
                    [
                        Column(ColumnNameBuilder.create("MIN", target_table_name), ColumnType("DOUBLE")),
                        Column(ColumnNameBuilder.create("RANGE", target_table_name), ColumnType("DOUBLE"))
                    ]),
        purpose="MIN_AND_RANGE_TABLE"
    )
    return [expected_parameter_table]


def test_min_max_scaler_create_from_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableNameBuilder.create("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLMinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    from_clause_part = scaler.create_transform_from_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    assert from_clause_part[0] == textwrap.dedent(
        f'''CROSS JOIN "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS" AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS"''')


def test_min_max_scaler_create_select_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableNameBuilder.create("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLMinMaxScaler()
    mock_sql_executor = MockSQLExecutor()
    select_clause_parts = scaler.create_transform_select_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    expected = textwrap.dedent(
        '''("IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"-"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS"."MIN")/"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS"."RANGE" AS "SRC_COLUMN1_MIN_MAX_SCALED"''')
    expected_transformation_column = \
        TransformationColumn(
            source_column=ColumnNameBuilder.create("SRC_COLUMN1", source_table),
            input_column=ColumnNameBuilder.create("SRC_COLUMN1", input_table),
            purpose="MinMaxScaled",
            column=Column(ColumnNameBuilder.create("SRC_COLUMN1_MIN_MAX_SCALED"), ColumnType("DOUBLE"))
        )
    assert len(select_clause_parts) == 1
    assert select_clause_parts[0].select_clause_part_expression == expected
    assert select_clause_parts[0].tranformation_column == expected_transformation_column

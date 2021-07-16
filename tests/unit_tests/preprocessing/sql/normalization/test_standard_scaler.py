import textwrap

from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_standard_scaler import SQLStandardScaler
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table import Table
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from tests.unit_tests.mock_sql_executor import MockSQLExecutor


def test_standard_scaler_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLStandardScaler()
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = scaler.fit(mock_sql_executor, source_column, target_schema, experiment_name)

    expected_parameter_tables = get_expected_parameter_tables()

    expected = textwrap.dedent("""
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS" AS
        SELECT
            CAST(AVG("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "AVG",
            CAST(STDDEV_POP("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "STDDEV"
        FROM "SRC_SCHEMA"."SRC_TABLE"
        """)
    assert mock_sql_executor.queries == [expected]
    assert expected_parameter_tables == parameter_tables


def get_expected_parameter_tables():
    target_table_name = TableName("EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS",
                                  SchemaName("TGT_SCHEMA"))
    expected_parameter_table = ParameterTable(
        source_column=ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
        table=Table(target_table_name,
                    columns=
                    [
                        Column(ColumnName("AVG", target_table_name), ColumnType("DOUBLE")),
                        Column(ColumnName("STDDEV", target_table_name), ColumnType("DOUBLE"))
                    ]),
        purpose="AVG_AND_STDDEV_TABLE"
    )
    return [expected_parameter_table]


def test_standard_scaler_create_from_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLStandardScaler()
    mock_sql_executor = MockSQLExecutor()
    from_clause_part = scaler.create_transform_from_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    assert from_clause_part[0] == textwrap.dedent(
        f'''CROSS JOIN "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS" AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS"''')


def test_standard_scaler_create_select_clause_part():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableName("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column = ColumnName("SRC_COLUMN1", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableName("IN_TABLE", input_schema)
    experiment_name = ExperimentName("EXPERIMENT")
    scaler = SQLStandardScaler()
    mock_sql_executor = MockSQLExecutor()
    select_clause_parts = scaler.create_transform_select_clause_part(
        mock_sql_executor, source_column, input_table, target_schema, experiment_name)
    expected = textwrap.dedent(
        '''("IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"-"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS"."AVG")/
        (CASE 
        WHEN "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS"."STDDEV" = 0 THEN 1
        ELSE "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS"."STDDEV"
        END)
        AS "SRC_COLUMN1_STANDARD_SCALED"''')
    expected_transformation_column = \
        TransformationColumn(
            source_column=ColumnName("SRC_COLUMN1", source_table),
            input_column=ColumnName("SRC_COLUMN1", input_table),
            purpose="StandardScaled",
            column=Column(ColumnName("SRC_COLUMN1_STANDARD_SCALED"), ColumnType("DOUBLE"))
        )
    assert len(select_clause_parts) == 1
    assert select_clause_parts[0].select_clause_part_expression == expected
    assert select_clause_parts[0].tranformation_column == expected_transformation_column

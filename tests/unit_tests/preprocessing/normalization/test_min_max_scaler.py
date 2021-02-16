import textwrap

from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler


def test_min_max_scaler_create_fit_queries():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    scaler = MinMaxScaler()
    queries = scaler.create_fit_queries(source_schema, source_table, source_column, target_schema)
    expected = textwrap.dedent("""
        CREATE OR REPLACE TABLE "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" AS
        SELECT
            MIN("SOURCE_SCHEMA"."SOURCE_TABLE"."SOURCE_COLUMN1") as "MIN",
            (MAX("SOURCE_SCHEMA"."SOURCE_TABLE"."SOURCE_COLUMN1")-MIN("SOURCE_SCHEMA"."SOURCE_TABLE"."SOURCE_COLUMN1")) as "RANGE"
        FROM "SOURCE_SCHEMA"."SOURCE_TABLE"
        """)
    assert queries == [expected]


def test_min_max_scaler_create_from_clause_part():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    input_schema = "INPUT_SCHEMA"
    input_table = "INPUT_TABLE"
    scaler = MinMaxScaler()
    from_clause_part = scaler.create_from_clause_part(source_schema, source_table, source_column,
                                                      input_schema, input_table,
                                                      target_schema)
    assert from_clause_part == [textwrap.dedent(f'''
        CROSS JOIN "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" 
        AS "TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"
        ''')]


def test_min_max_scaler_create_select_clause_part():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    input_schema = "INPUT_SCHEMA"
    input_table = "INPUT_TABLE"
    scaler = MinMaxScaler()
    select_clause_part = scaler.create_select_clause_part(source_schema, source_table, source_column,
                                                          input_schema, input_table,
                                                          target_schema)
    expected = textwrap.dedent('''
        (
            ("INPUT_SCHEMA"."INPUT_TABLE"."SOURCE_COLUMN1" -
                "TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."MIN") /
            "TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."RANGE"
        ) AS "SOURCE_COLUMN1_MIN_MAX_SCALED"''')
    assert select_clause_part == [expected]

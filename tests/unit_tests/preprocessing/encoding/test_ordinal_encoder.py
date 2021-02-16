import textwrap

from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder


def test_ordinal_encoder_create_fit_queries():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    encoder = OrdinalEncoder()
    queries = encoder.create_fit_queries(source_schema, source_table, source_column, target_schema)
    expected = textwrap.dedent(f"""
            CREATE OR REPLACE TABLE "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_ORDINAL_ENCODER_DICTIONARY" AS
            SELECT
                rownum - 1 as "ID",
                "SOURCE_COLUMN1" as "VALUE"
            FROM (
                SELECT distinct "SOURCE_COLUMN1"
                FROM "SOURCE_SCHEMA"."SOURCE_TABLE"
                ORDER BY "SOURCE_COLUMN1"
            );
            """)
    assert queries == [expected]


def test_ordinal_encoder_create_from_clause_part():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    input_schema = "INPUT_SCHEMA"
    input_table = "INPUT_TABLE"
    encoder = OrdinalEncoder()
    from_clause_part = encoder.create_from_clause_part(source_schema, source_table, source_column,
                                                       input_schema, input_table,
                                                       target_schema)
    expected = textwrap.dedent("""
            JOIN "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            AS "TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_ORDINAL_ENCODER_DICTIONARY"
            ON
                "TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."VALUE" = 
                "INPUT_SCHEMA"."INPUT_TABLE"."SOURCE_COLUMN1"
            """)
    assert from_clause_part == [expected]


def test_ordinal_encoder_create_select_clause_part():
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column = "SOURCE_COLUMN1"
    input_schema = "INPUT_SCHEMA"
    input_table = "INPUT_TABLE"
    encoder = OrdinalEncoder()
    select_clause_part = encoder.create_select_clause_part(source_schema, source_table, source_column,
                                                           input_schema, input_table,
                                                           target_schema)
    expected = textwrap.dedent(
        '"TARGET_SCHEMA_SOURCE_SCHEMA_SOURCE_TABLE_SOURCE_COLUMN1_ORDINAL_ENCODER_DICTIONARY"."ID" AS "SOURCE_COLUMN1_ID"')
    assert select_clause_part == [expected]

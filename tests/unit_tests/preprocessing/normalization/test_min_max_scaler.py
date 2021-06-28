import textwrap

from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table


def test_min_max_scaler_create_fit_queries():
    source_schema = Schema("SRC_SCHEMA")
    source_table = Table("SRC_TABLE", source_schema)
    target_schema = Schema("TGT_SCHEMA")
    source_column = Column("SRC_COLUMN1", source_table)
    scaler = MinMaxScaler()
    queries = scaler.create_fit_queries(source_column, target_schema)
    expected = textwrap.dedent("""
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" AS
        SELECT
            MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as "MIN",
            MAX("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1")-MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as "RANGE"
        FROM "SRC_SCHEMA"."SRC_TABLE"
        """)
    assert queries == [expected]


def test_min_max_scaler_create_from_clause_part():
    source_schema = Schema("SRC_SCHEMA")
    source_table = Table("SRC_TABLE", source_schema)
    target_schema = Schema("TGT_SCHEMA")
    source_column = Column("SRC_COLUMN1", source_table)
    input_schema = Schema("IN_SCHEMA")
    input_table = Table("IN_TABLE", input_schema)
    scaler = MinMaxScaler()
    from_clause_part = scaler.create_transform_from_clause_part(
        source_column, input_table, target_schema)
    assert from_clause_part[0] == textwrap.dedent(
        f'''CROSS JOIN "TGT_SCHEMA"."SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS" AS "TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"''')


def test_min_max_scaler_create_select_clause_part():
    source_schema = Schema("SRC_SCHEMA")
    source_table = Table("SRC_TABLE", source_schema)
    target_schema = Schema("TGT_SCHEMA")
    source_column = Column("SRC_COLUMN1", source_table)
    input_schema = Schema("IN_SCHEMA")
    input_table = Table("IN_TABLE", input_schema)
    scaler = MinMaxScaler()
    select_clause_part = scaler.create_transform_select_clause_part(
        source_column, input_table, target_schema)
    expected = textwrap.dedent(
        '''("IN_SCHEMA"."IN_TABLE"."SRC_COLUMN1"-"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."MIN")/"TGT_SCHEMA_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALAR_PARAMETERS"."RANGE" AS "SRC_COLUMN1_MIN_MAX_SCALED"''')
    assert select_clause_part == [expected]

import textwrap
from collections import OrderedDict

from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_transformer_creator import \
    ColumnTransformerCreator
from tests.unit_tests.mock_result_set import MockResultSet
from tests.unit_tests.mock_sql_executor import MockSQLExecutor


def test_happy_path():
    sql_executor = MockSQLExecutor(
        result_sets=[
            MockResultSet(rows=[
                ("a", "DECIMAL(18,0)"),
                ("b", "DOUBLE"),
                ("c", "VARCHAR(20000)"),
                ("d", "DOUBLE"),
            ]),
            MockResultSet(),
            MockResultSet(),
            MockResultSet(),
            MockResultSet(),
            MockResultSet(
                rows=[
                    (1,),
                    (2,),
                    (4,)
                ],
                columns={"VALUE": {"TYPE_NAME": "DECIMAL(18,0)"}}
            ),
            MockResultSet(
                rows=[
                    (1.0, 3.0),
                ],
                columns=OrderedDict(
                    [
                        ("MIN", {"TYPE_NAME": "DOUBLE"}),
                        ("RANGE", {"TYPE_NAME": "DOUBLE"})
                    ])
            ),
            MockResultSet(
                rows=[
                    ("1",),
                    ("2",),
                ],
                columns={"VALUE": {"TYPE_NAME": "VARCHAR(200000)"}}
            ),
            MockResultSet(
                rows=[
                    (1, 1.0, "1"),
                    (1, 1.0, "1"),
                ], ),
            MockResultSet(
                rows=[
                    (1.0, 3.0),
                ],
                columns=OrderedDict(
                    [
                        ("MIN", {"TYPE_NAME": "DOUBLE"}),
                        ("RANGE", {"TYPE_NAME": "DOUBLE"})
                    ])
            ),
            MockResultSet(
                rows=[
                    (1.0,),
                    (1.0,),
                ], ),
        ])
    creator = ColumnTransformerCreator()
    result = creator.generate_column_transformers(
        sql_executor=sql_executor,
        input_columns=[ColumnName("a"), ColumnName("b"), ColumnName("c")],
        target_columns=[ColumnName("d")],
        source_table=TableName("SRC_TABLE", SchemaName("SRC_SCHEMA")),
        target_schema=SchemaName("SRC_SCHEMA")
    )
    request_types_sql = f'''
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='SRC_SCHEMA'
            AND COLUMN_TABLE='SRC_TABLE'
            AND COLUMN_NAME in ('a','b','c','d')
            '''
    create_dictionary_a = textwrap.dedent('''
               CREATE OR REPLACE TABLE "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_a_ORDINAL_ENCODER_DICTIONARY" AS
               SELECT
                   CAST(rownum - 1 AS INTEGER) as "ID",
                   "VALUE"
               FROM (
                   SELECT DISTINCT "SRC_SCHEMA"."SRC_TABLE"."a" as "VALUE"
                   FROM "SRC_SCHEMA"."SRC_TABLE"
                   ORDER BY "SRC_SCHEMA"."SRC_TABLE"."a"
               );
                ''')
    create_b_min_max_parameter = textwrap.dedent('''
               CREATE OR REPLACE TABLE "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_b_MIN_MAX_SCALAR_PARAMETERS" AS
               SELECT
                   CAST(MIN("SRC_SCHEMA"."SRC_TABLE"."b") as DOUBLE) as "MIN",
                   CAST(MAX("SRC_SCHEMA"."SRC_TABLE"."b")-MIN("SRC_SCHEMA"."SRC_TABLE"."b") as DOUBLE) as "RANGE"
               FROM "SRC_SCHEMA"."SRC_TABLE"
                ''')
    create_c_dictionary = textwrap.dedent('''
               CREATE OR REPLACE TABLE "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_c_ORDINAL_ENCODER_DICTIONARY" AS
               SELECT
                   CAST(rownum - 1 AS INTEGER) as "ID",
                   "VALUE"
               FROM (
                   SELECT DISTINCT "SRC_SCHEMA"."SRC_TABLE"."c" as "VALUE"
                   FROM "SRC_SCHEMA"."SRC_TABLE"
                   ORDER BY "SRC_SCHEMA"."SRC_TABLE"."c"
               );
                ''')
    create_d_min_max = textwrap.dedent('''
               CREATE OR REPLACE TABLE "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_d_MIN_MAX_SCALAR_PARAMETERS" AS
               SELECT
                   CAST(MIN("SRC_SCHEMA"."SRC_TABLE"."d") as DOUBLE) as "MIN",
                   CAST(MAX("SRC_SCHEMA"."SRC_TABLE"."d")-MIN("SRC_SCHEMA"."SRC_TABLE"."d") as DOUBLE) as "RANGE"
               FROM "SRC_SCHEMA"."SRC_TABLE"
               ''')
    select_a_parameter = 'SELECT "VALUE" FROM "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_a_ORDINAL_ENCODER_DICTIONARY"'
    select_b_parameter = 'SELECT "MIN", "RANGE"  FROM "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_b_MIN_MAX_SCALAR_PARAMETERS"'
    select_c_parameter = 'SELECT "VALUE" FROM "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_c_ORDINAL_ENCODER_DICTIONARY"'
    select_a_b_c = 'SELECT "a","b","c" FROM "SRC_SCHEMA"."SRC_TABLE" LIMIT 2'
    select_d_parameter = 'SELECT "MIN", "RANGE"  FROM "SRC_SCHEMA"."SRC_SCHEMA_SRC_TABLE_d_MIN_MAX_SCALAR_PARAMETERS"'
    select_d = 'SELECT "d" FROM "SRC_SCHEMA"."SRC_TABLE" LIMIT 2'
    expected_queries = [request_types_sql,
                        create_dictionary_a,
                        create_b_min_max_parameter,
                        create_c_dictionary,
                        create_d_min_max,
                        select_a_parameter,
                        select_b_parameter,
                        select_c_parameter,
                        select_a_b_c,
                        select_d_parameter,
                        select_d]
    assert sql_executor.queries == expected_queries
    assert len(result.input_columns_transformer.transformers) == 3
    assert len(result.target_column_transformer.transformers) == 1

import pyexasol

from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.preprocessing.table_preprocessor import TablePreprocessor, \
    ColumnPreprocesserDefinition


def test_table_preprocessor_create_fit_queries():
    c = pyexasol.connect(dsn="localhost:8888", user="sys", password="exasol")

    c.execute("""
    CREATE SCHEMA IF NOT EXISTS "SOURCE_SCHEMA"
    """)
    c.execute("""
    CREATE SCHEMA IF NOT EXISTS "TARGET_SCHEMA"
    """)
    c.execute("""
    CREATE OR REPLACE TABLE "SOURCE_SCHEMA"."SOURCE_TABLE" (
        "CATEGORY" VARCHAR(1000),
        "NUMERICAL" FLOAT
    )
    """)
    c.execute("""INSERT INTO "SOURCE_SCHEMA"."SOURCE_TABLE" VALUES ('A',1),('B',2);""")
    source_schema = "SOURCE_SCHEMA"
    source_table = "SOURCE_TABLE"
    target_schema = "TARGET_SCHEMA"
    source_column1 = "CATEGORY"
    source_column2 = "NUMERICAL"
    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1, OrdinalEncoder()),
        ColumnPreprocesserDefinition(source_column2, MinMaxScaler()),
    ]

    table_preprocessor = TablePreprocessor(target_schema, source_schema, source_table, column_preprocessor_defintions)
    queries = table_preprocessor.create_fit_queries()
    for query in queries:
        c.execute(query)

    query = '''
    SELECT "ID", "VALUE"
    FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_CATEGORY_ORDINAL_ENCODER_DICTIONARY";
     '''
    result = c.execute(query).fetchall()
    assert result == [("0", "A"), ("1", "B")]

    query = '''
    SELECT "MIN", "RANGE"
    FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_NUMERICAL_MIN_MAX_SCALAR_PARAMETERS";
     '''
    result = c.execute(query).fetchall()
    assert result == [(1.0, 1.0)]

    query = table_preprocessor.create_transform_query(source_schema, source_table)
    print(query)
    c.execute(query)

    query = '''SELECT * FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_TRANSFORMED"'''
    result = c.execute(query).fetchall()
    assert result == [('0', 0.0), ('1', 1.0)]

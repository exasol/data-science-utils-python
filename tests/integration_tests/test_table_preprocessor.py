import pyexasol

from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor
from exasol_data_science_utils_python.preprocessing.schema.column_name import Column
from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema
from exasol_data_science_utils_python.preprocessing.schema.table_name import Table
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
    source_schema = Schema("SOURCE_SCHEMA")
    source_table = Table("SOURCE_TABLE", source_schema)
    target_schema = Schema("TARGET_SCHEMA")
    source_column1 = Column("CATEGORY", source_table)
    source_column2 = Column("NUMERICAL", source_table)
    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1.name, OrdinalEncoder()),
        ColumnPreprocesserDefinition(source_column2.name, MinMaxScaler()),
    ]

    sql_executor = PyexasolSQLExecutor(c)
    table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
    fit_tables = table_preprocessor.fit(sql_executor)

    query = '''
    SELECT "ID", "VALUE"
    FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_CATEGORY_ORDINAL_ENCODER_DICTIONARY";
     '''
    result = c.execute(query).fetchall()
    assert result == [(0, "A"), (1, "B")]

    query = '''
    SELECT "MIN", "RANGE"
    FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_NUMERICAL_MIN_MAX_SCALAR_PARAMETERS";
     '''
    result = c.execute(query).fetchall()
    assert result == [(1.0, 1.0)]

    transform_table = table_preprocessor.transform(sql_executor, source_table)

    query = '''SELECT * FROM "TARGET_SCHEMA"."SOURCE_SCHEMA_SOURCE_TABLE_TRANSFORMED"'''
    result = c.execute(query).fetchall()
    assert result == [(0, 0.0), (1, 1.0)]


def test_table_preprocessor_transform_queries():
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
    c.execute("""INSERT INTO "SOURCE_SCHEMA"."SOURCE_TABLE" VALUES ('A',1),('B',3);""")
    c.execute("""
    CREATE OR REPLACE TABLE "SOURCE_SCHEMA"."INPUT_TABLE" (
        "CATEGORY" VARCHAR(1000),
        "NUMERICAL" FLOAT
    )
    """)
    c.execute("""INSERT INTO "SOURCE_SCHEMA"."INPUT_TABLE" VALUES ('A',1),('B',3),('A',2),('C',4);""")

    source_schema = Schema("SOURCE_SCHEMA")
    source_table = Table("SOURCE_TABLE", source_schema)
    input_table = Table("INPUT_TABLE", source_schema)
    target_schema = Schema("TARGET_SCHEMA")
    source_column1 = Column("CATEGORY", source_table)
    source_column2 = Column("NUMERICAL", source_table)
    column_preprocessor_defintions = [
        ColumnPreprocesserDefinition(source_column1.name, OrdinalEncoder()),
        ColumnPreprocesserDefinition(source_column2.name, MinMaxScaler()),
    ]

    sql_executor = PyexasolSQLExecutor(c)
    table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
    fit_tables = table_preprocessor.fit(sql_executor)

    transform_table = table_preprocessor.transform(sql_executor, input_table)
    rs = c.execute(
        """select * from "TARGET_SCHEMA"."SOURCE_SCHEMA_INPUT_TABLE_TRANSFORMED" ORDER BY "NUMERICAL_MIN_MAX_SCALED" """)
    rows = rs.fetchall()
    assert rows == [(0, 0.0), (0, 0.5), (1, 1.0), (None, 1.5)]

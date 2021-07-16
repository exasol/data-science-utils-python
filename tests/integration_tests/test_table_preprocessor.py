import pandas as pd
import pyexasol

from exasol_data_science_utils_python.preprocessing.sql.encoding.sql_ordinal_encoder import SQLOrdinalEncoder
from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_min_max_scaler import SQLMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql.normalization.sql_standard_scaler import SQLStandardScaler
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor_definition import \
    SQLColumnPreprocessorDefinition
from exasol_data_science_utils_python.preprocessing.sql.sql_table_preprocessor import SQLTablePreprocessor
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor


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
    source_schema = SchemaName("SOURCE_SCHEMA")
    source_table = TableName("SOURCE_TABLE", source_schema)
    target_schema = SchemaName("TARGET_SCHEMA")
    source_column1 = ColumnName("CATEGORY", source_table)
    source_column2 = ColumnName("NUMERICAL", source_table)
    experiment_name = ExperimentName("EXPERIMENT")
    column_preprocessor_defintions = [
        SQLColumnPreprocessorDefinition(source_column1.name, SQLOrdinalEncoder()),
        SQLColumnPreprocessorDefinition(source_column2.name, SQLMinMaxScaler()),
        SQLColumnPreprocessorDefinition(source_column2.name, SQLStandardScaler()),
    ]

    sql_executor = PyexasolSQLExecutor(c)
    table_preprocessor = SQLTablePreprocessor(target_schema, source_table, experiment_name,
                                              column_preprocessor_defintions)
    fit_tables = table_preprocessor.fit(sql_executor)

    query = '''
    SELECT "ID", "VALUE"
    FROM "TARGET_SCHEMA"."EXPERIMENT_SOURCE_SCHEMA_SOURCE_TABLE_CATEGORY_ORDINAL_ENCODER_DICTIONARY";
     '''
    result = c.execute(query).fetchall()
    assert result == [(0, "A"), (1, "B")]

    query = '''
    SELECT "MIN", "RANGE"
    FROM "TARGET_SCHEMA"."EXPERIMENT_SOURCE_SCHEMA_SOURCE_TABLE_NUMERICAL_MIN_MAX_SCALER_PARAMETERS";
     '''
    result = c.execute(query).fetchall()
    assert result == [(1.0, 1.0)]

    query = '''
    SELECT "AVG", "STDDEV"
    FROM "TARGET_SCHEMA"."EXPERIMENT_SOURCE_SCHEMA_SOURCE_TABLE_NUMERICAL_STANDARD_SCALER_PARAMETERS";
     '''
    result = c.execute(query).fetchall()
    assert result == [(1.5, 0.7071067811865476)]

    transform_table = table_preprocessor.transform(sql_executor, source_table)

    query = '''SELECT * FROM "TARGET_SCHEMA"."EXPERIMENT_SOURCE_SCHEMA_SOURCE_TABLE_TRANSFORMED"'''
    result = c.execute(query).fetchall()
    assert result == [(0, 0.0, -0.7071067811865475), (1, 1.0, 0.7071067811865475)]


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

    source_schema = SchemaName("SOURCE_SCHEMA")
    source_table = TableName("SOURCE_TABLE", source_schema)
    input_table = TableName("INPUT_TABLE", source_schema)
    target_schema = SchemaName("TARGET_SCHEMA")
    source_column1 = ColumnName("CATEGORY", source_table)
    source_column2 = ColumnName("NUMERICAL", source_table)
    experiment_name = ExperimentName("EXPERIMENT")
    column_preprocessor_defintions = [
        SQLColumnPreprocessorDefinition(source_column1.name, SQLOrdinalEncoder()),
        SQLColumnPreprocessorDefinition(source_column2.name, SQLMinMaxScaler()),
        SQLColumnPreprocessorDefinition(source_column2.name, SQLStandardScaler()),
    ]

    sql_executor = PyexasolSQLExecutor(c)
    table_preprocessor = SQLTablePreprocessor(target_schema, source_table, experiment_name,
                                              column_preprocessor_defintions)
    fit_tables = table_preprocessor.fit(sql_executor)

    transform_table = table_preprocessor.transform(sql_executor, input_table)
    rs = c.execute(
        """select * from "TARGET_SCHEMA"."EXPERIMENT_SOURCE_SCHEMA_INPUT_TABLE_TRANSFORMED" ORDER BY "NUMERICAL_MIN_MAX_SCALED" """)
    rows = rs.fetchall()
    assert rows == [(0, 0.0, -0.7071067811865475), (0, 0.5, 0.0), (1, 1.0, 0.7071067811865475), (None, 1.5, 1.414213562373095)]

import textwrap
from collections import OrderedDict

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_one_hot_transformer import \
    SKLearnPrefittedOneHotTransformer
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.encoding.one_hot_encoder_factory import \
    OneHotEncoderFactory
from tests.unit_tests.mock_result_set import MockResultSet
from tests.unit_tests.mock_sql_executor import MockSQLExecutor


def test_happy_path():
    sql_executor = MockSQLExecutor(
        result_sets=[
            MockResultSet(),
            MockResultSet(
                rows=[
                    ("A", 1),
                    ("B", 0),
                    ("C", 2),
                    ("D", 3),
                    ("E", 4)
                ],
                columns=OrderedDict([
                    ("VALUE", {"TYPE_NAME": "VARCHAR(2000)"}),
                    ("ID", {"TYPE_NAME": "INTEGER"}),
                ])
            )
        ]
    )
    factory = OneHotEncoderFactory()
    source_column = Column(ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
                           ColumnType(name="INTEGER"))
    target_schema = SchemaName("TGT_SCHEMA")
    experiment_name = ExperimentName("EXPERIMENT")
    column_preprocessor = factory.create(sql_executor, source_column, target_schema, experiment_name)
    create_dictionary_query = textwrap.dedent('''
               CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY" AS
               SELECT
                   CAST(rownum - 1 AS INTEGER) as "ID",
                   "VALUE"
               FROM (
                   SELECT DISTINCT "SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1" as "VALUE"
                   FROM "SRC_SCHEMA"."SRC_TABLE"
                   ORDER BY "SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1"
               );
    ''')
    get_dictionary_query = 'SELECT "VALUE" FROM "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_ORDINAL_ENCODER_DICTIONARY"'
    assert sql_executor.queries[0] == create_dictionary_query
    assert sql_executor.queries[1] == get_dictionary_query
    assert column_preprocessor.source_column == source_column
    assert column_preprocessor.target_schema == target_schema
    assert isinstance(column_preprocessor.transformer, SKLearnPrefittedOneHotTransformer)
    X = pd.DataFrame(data=[
        ["A"],
        ["C"],
        ["B"],
        ["D"]
    ], columns=["C1"])
    X_ = column_preprocessor.transformer.transform(X["C1"])
    expected_X_ = np.array(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
        ])
    print(X_)
    print(expected_X_)
    assert np.array_equal(X_, expected_X_)

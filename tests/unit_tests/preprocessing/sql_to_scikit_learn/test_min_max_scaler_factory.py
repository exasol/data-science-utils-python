import textwrap

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scaler import SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.normalization.min_max_scaler_factory import \
    MinMaxScalerFactory
from exasol_data_science_utils_python.udf_utils.testing.mock_result_set import MockResultSet
from exasol_data_science_utils_python.udf_utils.testing.mock_sql_executor import MockSQLExecutor


def test_happy_path():
    sql_executor = MockSQLExecutor(
        result_sets=[
            MockResultSet(),
            MockResultSet(rows=[(1, 2)]),
        ]
    )
    source_column = Column(ColumnName("SRC_COLUMN1", TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
                           ColumnType(name="INTEGER"))
    target_schema = SchemaName("TGT_SCHEMA")
    experiment_name = ExperimentName("EXPERIMENT")

    factory = MinMaxScalerFactory()
    column_preprocessor = factory.create(sql_executor, source_column, target_schema, experiment_name)

    create_paraemter_table = textwrap.dedent(
        """
               CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS" AS
               SELECT
                   CAST(MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "MIN",
                   CAST(MAX("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1")-MIN("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "RANGE"
               FROM "SRC_SCHEMA"."SRC_TABLE"
        """
    )
    get_parameter_query = 'SELECT "MIN", "RANGE"  FROM "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_MIN_MAX_SCALER_PARAMETERS"'
    assert sql_executor.queries[0] == create_paraemter_table
    assert sql_executor.queries[1] == get_parameter_query
    assert column_preprocessor.source_column == source_column
    assert column_preprocessor.target_schema == target_schema
    assert isinstance(column_preprocessor.transformer, SKLearnPrefittedMinMaxScaler)
    X = pd.DataFrame(data=[
        [0],
        [1],
        [2],
        [3],
        [4],
        [1.5]
    ])
    X_ = column_preprocessor.transformer.transform(X[0])
    expected_X = pd.DataFrame(data=[[-0.5], [0.], [0.5], [1.], [1.5], [0.25]])[0]
    print(X_)
    print(expected_X)
    assert np.array_equal(X_, expected_X)

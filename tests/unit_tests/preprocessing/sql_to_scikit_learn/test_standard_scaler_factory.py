import textwrap

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_standard_scaler import \
    SKLearnPrefittedStandardScaler
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.normalization.standard_scaler_factory import \
    StandardScalerFactory
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.udf_utils.testing.mock_result_set import MockResultSet
from exasol_data_science_utils_python.udf_utils.testing.mock_sql_executor import MockSQLExecutor


def test_happy_path():
    sql_executor = MockSQLExecutor(
        result_sets=[
            MockResultSet(),
            MockResultSet(rows=[(1.5, 0.5)]),
        ]
    )
    source_column = Column(ColumnNameBuilder.create("SRC_COLUMN1",
                                      TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
                           ColumnType(name="INTEGER"))
    target_schema = SchemaName("TGT_SCHEMA")
    experiment_name = ExperimentName("EXPERIMENT")

    factory = StandardScalerFactory()
    column_preprocessor = factory.create(sql_executor, source_column, target_schema, experiment_name)

    create_paraemter_table = textwrap.dedent(
        """
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS" AS
        SELECT
            CAST(AVG("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "AVG",
            CAST(STDDEV_POP("SRC_SCHEMA"."SRC_TABLE"."SRC_COLUMN1") as DOUBLE) as "STDDEV"
        FROM "SRC_SCHEMA"."SRC_TABLE"
        """
    )
    get_parameter_query = \
        'SELECT "AVG", "STDDEV"  ' \
        'FROM "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_STANDARD_SCALER_PARAMETERS"'
    assert sql_executor.queries[0] == create_paraemter_table
    assert sql_executor.queries[1] == get_parameter_query
    assert column_preprocessor.source_column == source_column
    assert column_preprocessor.target_schema == target_schema
    assert isinstance(column_preprocessor.transformer, SKLearnPrefittedStandardScaler)
    X = pd.DataFrame(data=[
        [0],
        [1],
        [2],
        [3],
        [4],
        [1.5]
    ])
    X_ = column_preprocessor.transformer.transform(X[0])
    expected_X = pd.DataFrame(data=[[-3.0], [-1.0], [1.0], [3.0], [5.0], [0.0]])[0]
    print(X_)
    print(expected_X)
    assert np.array_equal(X_, expected_X)

from tempfile import TemporaryDirectory

import pandas as pd
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from numpy.random import RandomState
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF

    udf = PartialFitRegressorUDF(exa)

    def run(ctx: UDFContext):
        udf.run(ctx)


def test_partial_fit_regressor_udf():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("0", str, "VARCHAR(2000000)"),
            Column("1", str, "VARCHAR(2000000)"),
            Column("2", int, "INTEGER"),
            Column("3", int, "INTEGER"),
            Column("4", int, "INTEGER"),
            Column("5", float, "FLOAT"),
            Column("6", float, "FLOAT"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("output_model_path", str, "VARCHAR(2000000)"),
            Column("SCORE_SUM", float, "FLOAT"),
            Column("SCORE_COUNT", int, "INTEGER"),
        ]
    )
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=None)

        regressor_partial_fit_iterator = create_regressor_partial_fit_iterator()
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(
            regressor_partial_fit_iterator, PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)
        epochs = 10
        batch_size = 10
        shuffle_buffer_size = 30
        input_data = [
            (
                "MODEL_CONNECTION",
                "a,b",
                epochs,
                batch_size,
                shuffle_buffer_size,
                (1.0 * i) / 100,
                (1.0 * i) / 100
            )
            for i in range(100)
        ]
        result = list(executor.run([Group(input_data), Group(input_data)], exa))
        assert len(result) == 2
        for group in result:
            result_row = group.rows
            assert len(result_row) == 1
            print(result_row[0][0])
            # print(result_row[0][1] / result_row[0][2])


def create_regressor_partial_fit_iterator():
    input_preprocessor = ColumnTransformer(transformers=[
        ("a", MinMaxScaler(), ["a"]),
    ])
    input_preprocessor.fit(pd.DataFrame.from_dict({"a": [0.0, 1.0]}))
    output_preprocessor = ColumnTransformer(transformers=[
        ("b", MinMaxScaler(), ["b"]),
    ])
    output_preprocessor.fit(pd.DataFrame.from_dict({"b": [0.0, 1.0]}))
    model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
    regressor_partial_fit_iterator = RegressorPartialFitIterator(
        input_preprocessor=input_preprocessor,
        output_preprocessor=output_preprocessor,
        model=model
    )
    return regressor_partial_fit_iterator

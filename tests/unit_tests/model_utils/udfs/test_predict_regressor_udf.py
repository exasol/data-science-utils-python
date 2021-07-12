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
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_data_science_utils_python.model_utils.udfs.predict_regressor_udf import PredictRegressorUDF
    udf = PredictRegressorUDF(exa)

    def run(ctx: UDFContext):
        udf.run(ctx)


def test_score_regressor_udf():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("0", str, "VARCHAR(2000000)"),
            Column("1", str, "VARCHAR(2000000)"),
            Column("2", str, "VARCHAR(2000000)"),
            Column("3", int, "INTEGER"),
            Column("4", float, "FLOAT"),
            Column("5", int, "INTEGER"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("a", float, "FLOAT"),
            Column("id", int, "INTEGER"),
            Column("predicted_result", float, "FLOAT"),
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
        model_path = "model_to_score.pkl"
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(
            regressor_partial_fit_iterator, model_path)
        batch_size = 10
        input_data = [
            (
                "MODEL_CONNECTION",
                model_path,
                "a,id",
                batch_size,
                (1.0 * i) / 100,
                i
            )
            for i in range(100)
        ]
        result = list(executor.run([Group(input_data), Group(input_data)], exa))
        assert len(result) == 2
        for group in result:
            for row in group.rows:
                print(row)


def create_regressor_partial_fit_iterator():
    input_preprocessor = ColumnTransformer(transformers=[
        ("a", MinMaxScaler(), ["a"]),
    ])
    X = pd.DataFrame.from_dict({"a": [(1.0 * i) / 100 for i in range(100)]})
    input_preprocessor.fit(X)
    output_preprocessor = ColumnTransformer(transformers=[
        ("b", MinMaxScaler(), ["b"]),
    ])
    y = pd.DataFrame.from_dict({"b": [(1.0 * i) / 100 for i in range(100)]})
    output_preprocessor.fit(y)
    model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
    model.fit(X, y)
    regressor_partial_fit_iterator = RegressorPartialFitIterator(
        input_preprocessor=input_preprocessor,
        output_preprocessor=output_preprocessor,
        model=model
    )
    return regressor_partial_fit_iterator

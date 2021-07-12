from pathlib import PurePosixPath
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
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_data_science_utils_python.model_utils.udfs.combine_to_voting_regressor_udf import \
        CombineToVotingRegressorUDF

    udf = CombineToVotingRegressorUDF(exa)

    def run(ctx: UDFContext):
        udf.run(ctx)


def test_combine_to_voting_regressor_udf():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("input_model_path", str, "VARCHAR(2000000)"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("combined_model_path", str, "VARCHAR(2000000)"),
        ]
    )
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        path_under_model_connection = "my_path_under_model_connection"
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=path_under_model_connection)

        regressor_partial_fit_iterator = create_regressor_partial_fit_iterator()
        input_data = []
        for i in range(10):
            model_path = PurePosixPath(PartialFitRegressorUDF.BASE_MODEL_DIRECTORY, f"model{i}.pkl")
            model_bucketfs_location.upload_object_to_bucketfs_via_joblib(
                regressor_partial_fit_iterator, model_path)
            input_data.append(
                (
                    "MODEL_CONNECTION",
                    path_under_model_connection,
                    str(model_path)
                )
            )
        result = list(executor.run([Group(input_data)], exa))
        for i, group in enumerate(result):
            result_row = group.rows
            assert len(result_row) == 1
            model_connection_name = result_row[0][0]
            assert model_connection_name == "MODEL_CONNECTION"
            path_under_model_connection = result_row[0][1]
            assert path_under_model_connection == "my_path_under_model_connection"
            path_to_model = result_row[0][2]
            assert path_to_model == f"combined_models/123456789_123456789_0_123_{i}.pkl"
            output_model_bucketfs_location = \
                bucket_fs_factory.create_bucketfs_location(
                    url=model_connection.address,
                    user=model_connection.user,
                    pwd=model_connection.password,
                    base_path=path_under_model_connection)
            combined_score_iterator = \
                output_model_bucketfs_location.download_object_from_bucketfs_via_joblib(path_to_model)
            assert isinstance(combined_score_iterator, ScoreIterator)
            assert isinstance(combined_score_iterator.model, VotingRegressor)
            assert len(combined_score_iterator.model.estimators) == 10


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
    X = pd.DataFrame.from_dict({"a": [0.0, 1.0]})
    y = pd.DataFrame.from_dict({"b": [0.0, 1.0]})
    model.fit(X, y)
    regressor_partial_fit_iterator = RegressorPartialFitIterator(
        input_preprocessor=input_preprocessor,
        output_preprocessor=output_preprocessor,
        model=model
    )
    return regressor_partial_fit_iterator

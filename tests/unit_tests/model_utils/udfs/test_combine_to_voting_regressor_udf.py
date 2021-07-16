import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from numpy.random import RandomState
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scaler import \
    SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
    ColumnSetPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.udf_utils.abstract_bucketfs_location import AbstractBucketFSLocation
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
    meta = create_mock_metadata()
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location, path_under_model_connection = \
            create_model_bucketfs_location(bucket_fs_factory, model_connection)

        regressor_partial_fit_iterator = create_regressor_partial_fit_iterator()
        input_data, model_paths = create_input_data(path_under_model_connection, 1)
        upload_models(model_bucketfs_location, model_paths, regressor_partial_fit_iterator)

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
                    base_path=PurePosixPath(path_under_model_connection))
            combined_score_iterator = \
                output_model_bucketfs_location.download_object_from_bucketfs_via_joblib(path_to_model)
            assert isinstance(combined_score_iterator, ScoreIterator)
            assert isinstance(combined_score_iterator.model, VotingRegressor)
            assert len(combined_score_iterator.model.estimators) == 10


def test_combine_to_voting_regressor_udf_retry():
    executor = UDFMockExecutor()
    meta = create_mock_metadata()
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location, path_under_model_connection = \
            create_model_bucketfs_location(bucket_fs_factory, model_connection)

        regressor_partial_fit_iterator = create_regressor_partial_fit_iterator()
        input_data, model_paths = create_input_data(path_under_model_connection, 60)

        def run():
            result = list(executor.run([Group(input_data)], exa))
            return result

        threadpool = ThreadPoolExecutor()
        future = threadpool.submit(run)
        time.sleep(5)
        upload_models(model_bucketfs_location, model_paths, regressor_partial_fit_iterator)

        result = future.result()
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
                    base_path=PurePosixPath(path_under_model_connection))
            combined_score_iterator = \
                output_model_bucketfs_location.download_object_from_bucketfs_via_joblib(path_to_model)
            assert isinstance(combined_score_iterator, ScoreIterator)
            assert isinstance(combined_score_iterator.model, VotingRegressor)
            assert len(combined_score_iterator.model.estimators) == 10


def test_combine_to_voting_regressor_udf_retry_fails():
    executor = UDFMockExecutor()
    meta = create_mock_metadata()
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location, path_under_model_connection = \
            create_model_bucketfs_location(bucket_fs_factory, model_connection)

        input_data, model_paths = create_input_data(path_under_model_connection, 1)

        def run():
            result = list(executor.run([Group(input_data)], exa))
            return result

        threadpool = ThreadPoolExecutor()
        future = threadpool.submit(run)

        with pytest.raises(Exception) as c:
            result = future.result()


def create_model_bucketfs_location(bucket_fs_factory: BucketFSFactory, model_connection: Connection):
    path_under_model_connection = PurePosixPath("my_path_under_model_connection")
    model_bucketfs_location = \
        bucket_fs_factory.create_bucketfs_location(
            url=model_connection.address,
            user=model_connection.user,
            pwd=model_connection.password,
            base_path=path_under_model_connection)
    return model_bucketfs_location, path_under_model_connection


def create_input_data(path_under_model_connection: PurePosixPath, download_retry_seconds: int):
    input_data = []
    model_paths = []
    for i in range(10):
        model_path = PurePosixPath(PartialFitRegressorUDF.BASE_MODEL_DIRECTORY, f"model{i}.pkl")
        model_paths.append(model_path)
        input_data.append(
            (
                "MODEL_CONNECTION",
                str(path_under_model_connection),
                str(model_path),
                download_retry_seconds
            )
        )
    return input_data, model_paths


def upload_models(model_bucketfs_location: AbstractBucketFSLocation,
                  model_paths: List[PurePosixPath],
                  regressor_partial_fit_iterator):
    for model_path in model_paths:
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(
            regressor_partial_fit_iterator, model_path)


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("input_model_path", str, "VARCHAR(2000000)"),
            Column("download_retry_seconds", int, "INTEGER"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("combined_model_path", str, "VARCHAR(2000000)"),
        ]
    )
    return meta


def create_regressor_partial_fit_iterator():
    input_preprocessor = SKLearnPrefittedColumnTransformer(transformer_mapping=[
        ("a", SKLearnPrefittedMinMaxScaler(min_value=0, range_value=100)),
    ])
    output_preprocessor = SKLearnPrefittedColumnTransformer(transformer_mapping=[
        ("b", SKLearnPrefittedMinMaxScaler(min_value=0, range_value=100)),
    ])
    table_preprocessor = TablePreprocessor(
        input_column_set_preprocessors=ColumnSetPreprocessor(
            column_transformer=input_preprocessor,
        ),
        target_column_set_preprocessors=ColumnSetPreprocessor(
            column_transformer=output_preprocessor,
        ),
    )
    model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
    X = pd.DataFrame.from_dict({"a": [(1.0 * i) / 100 for i in range(100)]})
    y = pd.DataFrame.from_dict({"b": [(1.0 * i) / 100 for i in range(100)]})
    model.fit(X, y)
    iterator = RegressorPartialFitIterator(
        table_preprocessor=table_preprocessor,
        model=model
    )
    return iterator

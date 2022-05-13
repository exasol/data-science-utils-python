from tempfile import TemporaryDirectory

import pandas as pd
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from numpy.random import RandomState
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scaler import \
    SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
    ColumnSetPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory


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
        output_type="EMITS",
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
    input_preprocessor = SKLearnPrefittedColumnTransformer(transformer_mapping=[
        ("a", SKLearnPrefittedMinMaxScaler(min_value=0, range_value=100)),
    ])
    table_preprocessor = TablePreprocessor(
        input_column_set_preprocessors=ColumnSetPreprocessor(
            column_transformer=input_preprocessor,
        ),
    )
    model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
    X = pd.DataFrame.from_dict({"a": [(1.0 * i) / 100 for i in range(100)]})
    y = pd.DataFrame.from_dict({"b": [(1.0 * i) / 100 for i in range(100)]})
    model.fit(X, y)
    iterator = PredictionIterator(
        table_preprocessor=table_preprocessor,
        model=model
    )
    return iterator

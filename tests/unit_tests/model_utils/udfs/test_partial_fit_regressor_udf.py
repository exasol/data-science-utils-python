from pathlib import PurePosixPath
from tempfile import TemporaryDirectory

from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from numpy.random import RandomState
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scalar import \
    SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
    ColumnSetPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
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
            # Config
            Column("0", str, "VARCHAR(2000000)"),
            Column("1", str, "VARCHAR(2000000)"),
            Column("2", str, "VARCHAR(2000000)"),
            Column("3", int, "INTEGER"),
            Column("4", int, "INTEGER"),
            Column("5", int, "INTEGER"),
            # Data
            Column("6", float, "FLOAT"),
            Column("7", float, "FLOAT"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("model_connection_name", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("output_model_path", str, "VARCHAR(2000000)"),
            Column("training_score_sum", float, "FLOAT"),
            Column("training_score_count", int, "INTEGER"),
        ]
    )
    with TemporaryDirectory() as path:
        model_connection = Connection(address=f"file://{path}/model")
        exa = MockExaEnvironment(meta,
                                 connections={
                                     "MODEL_CONNECTION": model_connection
                                 })
        bucket_fs_factory = BucketFSFactory()
        path_under_model_connection = PurePosixPath("my_path_under_model_connection")
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=path_under_model_connection)

        regressor_partial_fit_iterator = create_regressor_partial_fit_iterator()
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(
            regressor_partial_fit_iterator, PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)
        epochs = 10
        batch_size = 10
        shuffle_buffer_size = 30
        input_data = [
            (
                "MODEL_CONNECTION",
                str(path_under_model_connection),
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
        for i, group in enumerate(result):
            result_row = group.rows
            assert len(result_row) == 1
            model_connection_name = result_row[0][0]
            assert model_connection_name == "MODEL_CONNECTION"
            path_under_model_connection = result_row[0][1]
            assert path_under_model_connection == "my_path_under_model_connection"
            path_to_model = result_row[0][2]
            assert path_to_model == f"base_models/123456789_123456789_0_123_{i}.pkl"
            output_model_bucketfs_location = \
                bucket_fs_factory.create_bucketfs_location(
                    url=model_connection.address,
                    user=model_connection.user,
                    pwd=model_connection.password,
                    base_path=PurePosixPath(path_under_model_connection))
            model = output_model_bucketfs_location.download_object_from_bucketfs_via_joblib(path_to_model)
            assert isinstance(model, RegressorPartialFitIterator)
            print(result_row[0][3] / result_row[0][4])


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
    iterator = RegressorPartialFitIterator(
        table_preprocessor=table_preprocessor,
        model=model
    )
    return iterator

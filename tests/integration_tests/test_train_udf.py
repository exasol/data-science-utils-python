import textwrap
from pathlib import Path

import pyexasol
import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


@pytest.fixture(scope="session")
def db_connection():
    db_connection = Connection(address=f"localhost:8888", user="sys", password="exasol")
    return db_connection


@pytest.fixture(scope="session")
def pyexasol_connection(db_connection):
    conn = pyexasol.connect(dsn=db_connection.address, user=db_connection.user, password=db_connection.password)
    return conn


@pytest.fixture(scope="session")
def upload_language_container(pyexasol_connection, language_container):
    container_connection = Connection(address=f"http://localhost:6583/default/container;bfsdefault",
                                      user="w", password="write")
    bucket_fs_factory = BucketFSFactory()
    container_bucketfs_location = \
        bucket_fs_factory.create_bucketfs_location(
            url=container_connection.address,
            user=container_connection.user,
            pwd=container_connection.password,
            base_path=None)
    container_path = Path(language_container["container_path"])
    alter_session = Path(language_container["alter_session"])
    pyexasol_connection.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{alter_session}'")
    pyexasol_connection.execute(f"ALTER SESSION SET SCRIPT_LANGUAGES='{alter_session}'")
    with open(container_path, "rb") as container_file:
        container_bucketfs_location.upload_fileobj_to_bucketfs(container_file, "ml.tar")


@pytest.fixture(scope="session")
def create_input_table(pyexasol_connection):
    pyexasol_connection.execute("""
        CREATE OR REPLACE TABLE TEST.ABC(
            P1 INTEGER,
            P2 INTEGER,
            A FLOAT,
            B FLOAT,
            C FLOAT
        )
        """)
    for i in range(1, 100):
        if i % 100 == 0:
            print(f"Insert {i}")
        values = ",".join([f"({j % 2},{i % 2},{j * 1.0 * i}, {j * 2.0 * i}, {j * 3.0 * i})" for j in range(1, 100)])
        pyexasol_connection.execute(f"INSERT INTO TEST.ABC VALUES {values}")
    print("COUNT", pyexasol_connection.execute("SELECT count(*) FROM TEST.ABC").fetchall())


def drop_and_create_target_schema(pyexasol_connection):
    try:
        pyexasol_connection.execute("""
    DROP SCHEMA TARGET_SCHEMA CASCADE;
    """)
    except:
        pass
    pyexasol_connection.execute("""CREATE SCHEMA TARGET_SCHEMA;""")


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from sklearn.linear_model import SGDRegressor
    from numpy.random import RandomState
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_description_based_table_preprocessor_factory import \
        ColumnDescriptionBasedTablePreprocessorFactory
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_description import \
        ColumnPreprocessorDescription
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.exact_column_name_selector import \
        ExactColumnNameSelector
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.normalization.min_max_scaler_factory import \
        MinMaxScalerFactory

    from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regression_train_udf import PartialFitRegressionTrainUDF

    train_udf = PartialFitRegressionTrainUDF()

    def run(ctx: UDFContext):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False,
                             fit_intercept=True, eta0=0.9, power_t=0.1, learning_rate='invscaling')
        table_preprocessor_factory = ColumnDescriptionBasedTablePreprocessorFactory(
            input_column_preprocessor_descriptions=[
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("A"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("B"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
            ],
            target_column_preprocessor_descriptions=[
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("C"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
            ]
        )
        train_udf.run(exa, ctx, model, table_preprocessor_factory)


def test_train_udf_with_mock_random_partitions(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 3
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=False,
            number_of_random_partitions=3,
            split_by_columns=None,
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_split_by_node(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 1
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=True,
            number_of_random_partitions=None,
            split_by_columns=None,
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_split_by_columns(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 4
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=False,
            number_of_random_partitions=None,
            split_by_columns="P1,P2",
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_random_partitions_and_split_by_columns(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 6
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=False,
            number_of_random_partitions=3,
            split_by_columns="P1",
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_split_by_node_and_random_partitions(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 2
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=True,
            number_of_random_partitions=2,
            split_by_columns=None
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_split_by_columns_empty_string(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    expected_number_of_base_models = 2
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=False,
            number_of_random_partitions=2,
            split_by_columns="",
        )
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len(unique_base_models) == expected_number_of_base_models
    assert len(fitted_combined_models) == 1
    assert len(result) == 1
    for group in result:
        assert len(group.rows) == 1


def test_train_udf_with_mock_multiple_groups(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    number_of_groups = 2
    expected_number_of_base_models = 2
    result, fitted_base_models, fitted_combined_models, unique_base_models = \
        run_mock_test_valid(
            db_connection,
            pyexasol_connection,
            split_by_node=False,
            number_of_random_partitions=2,
            split_by_columns="",
            number_of_groups=number_of_groups
        )
    unique_model_id_in_base_models = {row[1] for row in fitted_base_models}
    assert len(fitted_base_models) == expected_number_of_base_models * number_of_groups
    assert len(unique_model_id_in_base_models) == number_of_groups
    assert len(unique_base_models) == expected_number_of_base_models * number_of_groups
    assert len(fitted_combined_models) == 1 * number_of_groups
    assert len(result) == number_of_groups
    for group in result:
        assert len(group.rows) == 1


def run_mock_test_valid(db_connection,
                        pyexasol_connection,
                        split_by_node: bool,
                        number_of_random_partitions: int,
                        split_by_columns: str,
                        number_of_groups: int = 1):
    result = run_mock_test(db_connection,
                           pyexasol_connection,
                           split_by_node,
                           number_of_random_partitions,
                           split_by_columns,
                           number_of_groups)
    fitted_base_models, fitted_combined_models, unique_base_models = get_results(pyexasol_connection, result)
    return result, fitted_base_models, fitted_combined_models, unique_base_models


def get_results(pyexasol_connection, result):
    fitted_base_models = pyexasol_connection.execute("""
        SELECT * FROM TARGET_SCHEMA.FITTED_BASE_MODELS""").fetchall()
    print("fitted_base_models", fitted_base_models)
    fitted_combined_models = pyexasol_connection.execute("""
        SELECT * FROM TARGET_SCHEMA.FITTED_COMBINED_MODEL""").fetchall()
    print("fitted_combined_models", fitted_combined_models)
    unique_base_models = {row[4] for row in fitted_base_models}
    print("result", result)
    return fitted_base_models, fitted_combined_models, unique_base_models


def run_mock_test(db_connection,
                  pyexasol_connection,
                  split_by_node: bool,
                  number_of_random_partitions: int,
                  split_by_columns: str,
                  number_of_groups: int = 1):
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("download_retry_seconds", int, "INTEGER"),
            Column("db_connection", str, "VARCHAR(2000000)"),
            Column("source_schema_name", str, "VARCHAR(2000000)"),
            Column("source_table_name", str, "VARCHAR(2000000)"),
            Column("columns", str, "VARCHAR(2000000)"),
            Column("target_schema_name", str, "VARCHAR(2000000)"),
            Column("experiment_name", str, "VARCHAR(2000000)"),
            Column("epochs", int, "INTEGER"),
            Column("batch_size", int, "INTEGER"),
            Column("shuffle_buffer_size", int, "INTEGER"),
            Column("split_per_node", bool, "BOOLEAN"),
            Column("number_of_random_partitions", int, "INTEGER"),
            Column("split_by_columns", str, "VARCHAR(2000000)"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("job_id", str, "VARCHAR(2000000)"),
            Column("model_id", str, "VARCHAR(2000000)"),
            Column("model_connection_name", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("model_path", str, "VARCHAR(2000000)"),
        ]
    )
    model_connection, model_connection_name = \
        create_model_connection(pyexasol_connection)
    drop_and_create_target_schema(pyexasol_connection)
    exa = MockExaEnvironment(meta,
                             connections={
                                 "MODEL_CONNECTION": model_connection,
                                 "DB_CONNECTION": db_connection
                             })

    groups = [Group([(
        model_connection_name,
        "my_path_under_model_connection_" + str(i),
        60,
        "DB_CONNECTION",
        "TEST",
        "ABC",
        "A,B,C",
        "TARGET_SCHEMA",
        "EXPERIMENT",
        10,
        100,
        10000,
        split_by_node,
        number_of_random_partitions,
        split_by_columns
    )]) for i in range(number_of_groups)]
    result = list(executor.run(groups, exa))
    return result


def test_train_udf(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    model_connection, model_connection_name = \
        create_model_connection(pyexasol_connection)
    db_connection, db_connection_name = \
        create_db_connection(pyexasol_connection, db_connection)
    target_schema = SchemaName("TARGET_SCHEMA")
    drop_and_create_target_schema(pyexasol_connection)
    udf_sql = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3_DSUP SET SCRIPT {target_schema.fully_qualified()}."TRAIN_UDF"(
        model_connection VARCHAR(2000000),
        path_under_model_connection VARCHAR(2000000),
        download_retry_seconds INTEGER,
        db_connection VARCHAR(2000000),
        source_schema_name VARCHAR(2000000),
        source_table_name VARCHAR(2000000),
        columns VARCHAR(2000000),
        target_schema_name VARCHAR(2000000),
        experiment_name VARCHAR(2000000),
        epochs INTEGER,
        batch_size INTEGER,
        shuffle_buffer_size INTEGER,
        split_per_node BOOLEAN,
        number_of_random_partitions INTEGER,
        split_by_columns VARCHAR(2000000)
    ) 
    EMITS (
        job_id VARCHAR(2000000),
        model_id VARCHAR(2000000),
        model_connection_name VARCHAR(2000000),
        path_under_model_connection VARCHAR(2000000),
        model_path VARCHAR(2000000)
    ) AS
    from sklearn.linear_model import SGDRegressor
    from numpy.random import RandomState
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_description_based_table_preprocessor_factory import \
        ColumnDescriptionBasedTablePreprocessorFactory
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_description import \
        ColumnPreprocessorDescription
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.exact_column_name_selector import \
        ExactColumnNameSelector
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.normalization.min_max_scaler_factory import \
        MinMaxScalerFactory

    from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regression_train_udf import PartialFitRegressionTrainUDF

    train_udf = PartialFitRegressionTrainUDF()

    def run(ctx):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False,
                             fit_intercept=True, eta0=0.9, power_t=0.1, learning_rate='invscaling')
        table_preprocessor_factory = ColumnDescriptionBasedTablePreprocessorFactory(
            input_column_preprocessor_descriptions=[
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("A"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("B"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
            ],
            target_column_preprocessor_descriptions=[
                ColumnPreprocessorDescription(
                    column_selector=ExactColumnNameSelector("C"),
                    column_preprocessor_factory=MinMaxScalerFactory()
                ),
            ]
        )
        train_udf.run(exa, ctx, model, table_preprocessor_factory)
    """)
    pyexasol_connection.execute(udf_sql)
    query_udf = f"""
    select {target_schema.fully_qualified()}."TRAIN_UDF"(
        '{model_connection_name}',
        'my_path_under_model_connection',
        60,
        '{db_connection_name}',
        'TEST',
        'ABC',
        'A,B,C',
        'TARGET_SCHEMA',
        'EXPERIMENT',
        10,
        100,
        10000,
        True,
        4,
        null
    )
    """
    pyexasol_connection.execute(query_udf)
    fitted_base_models = pyexasol_connection.execute("""
    SELECT * FROM TARGET_SCHEMA.FITTED_BASE_MODELS""").fetchall()
    print(fitted_base_models)
    assert len(fitted_base_models) == 4
    fitted_combined_models = pyexasol_connection.execute("""
    SELECT * FROM TARGET_SCHEMA.FITTED_COMBINED_MODEL""").fetchall()
    print(fitted_combined_models)
    assert len(fitted_combined_models) == 1


def create_model_connection(conn):
    model_connection = Connection(address=f"http://localhost:6583/default/model;bfsdefault",
                                  user="w", password="write")
    model_connection_name = "MODEL_CONNECTION"
    return drop_and_create_connection(conn, model_connection, model_connection_name)


def create_db_connection(conn, db_connection):
    db_connection_name = "DB_CONNECTION"
    return drop_and_create_connection(conn, db_connection, db_connection_name)


def drop_and_create_connection(conn, model_connection, model_connection_name):
    try:
        conn.execute(f"DROP CONNECTION {model_connection_name}")
    except:
        pass
    conn.execute(
        f"CREATE CONNECTION {model_connection_name} TO '{model_connection.address}' USER '{model_connection.user}' IDENTIFIED BY '{model_connection.password}';")
    return model_connection, model_connection_name

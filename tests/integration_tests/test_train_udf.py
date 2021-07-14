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

from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
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
    from exasol_data_science_utils_python.model_utils.udfs.column_preprocessor_creator import ColumnPreprocessorCreator
    from exasol_data_science_utils_python.model_utils.udfs.train_udf import TrainUDF

    def run(ctx: UDFContext):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False,
                             fit_intercept=True, eta0=0.9, power_t=0.1, learning_rate='invscaling')
        column_preprocessor_creator = ColumnPreprocessorCreator()
        TrainUDF().run(exa, ctx, model, column_preprocessor_creator)


def test_train_udf_with_mock_random_partitions(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=False,
        number_of_random_partitions=3,
        split_by_columns=None,
        expected_number_of_base_models=3,
    )


def test_train_udf_with_mock_split_by_node(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=True,
        number_of_random_partitions=None,
        split_by_columns=None,
        expected_number_of_base_models=1,
    )


def test_train_udf_with_mock_split_by_columns(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=False,
        number_of_random_partitions=None,
        split_by_columns="P1,P2",
        expected_number_of_base_models=4,
    )


def test_train_udf_with_mock_random_partitions_and_split_by_columns(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=False,
        number_of_random_partitions=3,
        split_by_columns="P1",
        expected_number_of_base_models=6,
    )


def test_train_udf_with_mock_split_by_node_and_random_partitions(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=True,
        number_of_random_partitions=2,
        split_by_columns=None,
        expected_number_of_base_models=2,
    )


def test_train_udf_with_mock_split_by_columns_empty_string(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    run_mock_test_valid(
        db_connection,
        pyexasol_connection,
        split_by_node=False,
        number_of_random_partitions=2,
        split_by_columns="",
        expected_number_of_base_models=2,
    )


def run_mock_test_valid(db_connection,
                        pyexasol_connection,
                        split_by_node: bool,
                        number_of_random_partitions: int,
                        split_by_columns: str,
                        expected_number_of_base_models: int):
    run_mock_test(db_connection,
                  pyexasol_connection,
                  split_by_node,
                  number_of_random_partitions,
                  split_by_columns)
    fitted_base_models = pyexasol_connection.execute("""
        SELECT * FROM TARGET_SCHEMA.FITTED_BASE_MODELS""").fetchall()
    print(fitted_base_models)
    assert len(fitted_base_models) == expected_number_of_base_models
    assert len({row[3] for row in fitted_base_models}) == expected_number_of_base_models
    fitted_combined_models = pyexasol_connection.execute("""
        SELECT * FROM TARGET_SCHEMA.FITTED_COMBINED_MODEL""").fetchall()
    print(fitted_combined_models)
    assert len(fitted_combined_models) == 1


def run_mock_test(db_connection,
                  pyexasol_connection,
                  split_by_node: bool,
                  number_of_random_partitions: int,
                  split_by_columns: str):
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("path_under_model_connection", str, "VARCHAR(2000000)"),
            Column("db_connection", str, "VARCHAR(2000000)"),
            Column("source_schema_name", str, "VARCHAR(2000000)"),
            Column("source_table_name", str, "VARCHAR(2000000)"),
            Column("input_columns", str, "VARCHAR(2000000)"),
            Column("target_column", str, "VARCHAR(2000000)"),
            Column("target_schema_name", str, "VARCHAR(2000000)"),
            Column("epochs", int, "INTEGER"),
            Column("batch_size", int, "INTEGER"),
            Column("shuffle_buffer_size", int, "INTEGER"),
            Column("split_per_node", bool, "BOOLEAN"),
            Column("number_of_random_partitions", int, "INTEGER"),
            Column("split_by_columns", str, "VARCHAR(2000000)"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("output_model_path", str, "VARCHAR(2000000)"),
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
    input_data = []
    input_data.append(
        (
            model_connection_name,
            "my_path_under_model_connection",
            "DB_CONNECTION",
            "TEST",
            "ABC",
            "A,B",
            "C",
            "TARGET_SCHEMA",
            10,
            100,
            10000,
            split_by_node,
            number_of_random_partitions,
            split_by_columns
        )
    )
    result = list(executor.run([Group(input_data)], exa))


def test_train_udf(
        upload_language_container,
        create_input_table,
        pyexasol_connection,
        db_connection):
    model_connection, model_connection_name = \
        create_model_connection(pyexasol_connection)
    db_connection, db_connection_name = \
        create_db_connection(pyexasol_connection, db_connection)
    target_schema = Schema("TARGET_SCHEMA")
    drop_and_create_target_schema(pyexasol_connection)
    udf_sql = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3_DSUP SET SCRIPT {target_schema.fully_qualified()}."TRAIN_UDF"(
        model_connection VARCHAR(2000000),
        path_under_model_connection VARCHAR(2000000),
        db_connection VARCHAR(2000000),
        source_schema_name VARCHAR(2000000),
        source_table_name VARCHAR(2000000),
        input_columns VARCHAR(2000000),
        target_column VARCHAR(2000000),
        target_schema_name VARCHAR(2000000),
        epochs INTEGER,
        batch_size INTEGER,
        shuffle_buffer_size INTEGER,
        split_per_node BOOLEAN,
        number_of_random_partitions INTEGER,
        split_by_columns VARCHAR(2000000)
    ) 
    EMITS (o varchar(10000)) AS
    from sklearn.linear_model import SGDRegressor
    from numpy.random import RandomState
    from exasol_data_science_utils_python.model_utils.udfs.column_preprocessor_creator import ColumnPreprocessorCreator
    from exasol_data_science_utils_python.model_utils.udfs.train_udf import TrainUDF

    def run(ctx):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False,
                             fit_intercept=True, eta0=0.9, power_t=0.1, learning_rate='invscaling')
        column_preprocessor_creator = ColumnPreprocessorCreator()
        TrainUDF().run(exa, ctx, model, column_preprocessor_creator)
    """)
    pyexasol_connection.execute(udf_sql)
    query_udf = f"""
    select {target_schema.fully_qualified()}."TRAIN_UDF"(
        '{model_connection_name}',
        'my_path_under_model_connection',
        '{db_connection_name}',
        'TEST',
        'ABC',
        'A,B',
        'C',
        'TARGET_SCHEMA',
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

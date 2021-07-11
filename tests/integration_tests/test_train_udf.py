import os
from pathlib import Path

import pyexasol
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


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


def test_train_udf(language_container):
    print("language_container", language_container)
    print(os.getcwd())
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_connection", str, "VARCHAR(2000000)"),
            Column("db_connection", str, "VARCHAR(2000000)"),
            Column("source_schema_name", str, "VARCHAR(2000000)"),
            Column("source_table_name", str, "VARCHAR(2000000)"),
            Column("input_columns", str, "VARCHAR(2000000)"),
            Column("target_column", str, "VARCHAR(2000000)"),
            Column("target_schema_name", str, "VARCHAR(2000000)"),
            Column("epochs", int, "INTEGER"),
            Column("batch_size", int, "INTEGER"),
            Column("shuffle_buffer_size", int, "INTEGER"),
        ],
        output_type="EMIT",
        output_columns=[
            Column("output_model_path", str, "VARCHAR(2000000)"),
        ]
    )
    db_connection = Connection(address=f"localhost:8888", user="sys", password="exasol")
    bucket_fs_factory = BucketFSFactory()

    conn = pyexasol.connect(dsn=db_connection.address, user=db_connection.user, password=db_connection.password)
    create_input_table(conn)
    upload_language_container(conn, language_container)
    model_connection, model_connection_name = create_model_connection(conn)

    exa = MockExaEnvironment(meta,
                             connections={
                                 "MODEL_CONNECTION": model_connection,
                                 "DB_CONNECTION": db_connection
                             })
    model_bucketfs_location = \
        bucket_fs_factory.create_bucketfs_location(
            url=model_connection.address,
            user=model_connection.user,
            pwd=model_connection.password,
            base_path=None)

    input_data = []
    input_data.append(
        (
            model_connection_name,
            "DB_CONNECTION",
            "TEST",
            "ABC",
            "A,B",
            "C",
            "TARGET_SCHEMA",
            10,
            100,
            10000
        )
    )
    result = list(executor.run([Group(input_data)], exa))


def create_model_connection(conn):
    model_connection = Connection(address=f"http://localhost:6583/default/model;bfsdefault",
                                  user="w", password="write")
    model_connection_name = "MODEL_CONNECTION"
    try:
        conn.execute(f"DROP CONNECTION {model_connection_name}")
    except:
        pass
    conn.execute(
        f"CREATE CONNECTION {model_connection_name} TO 'http://localhost:6583/default/model;bfsdefault' USER '{model_connection.user}' IDENTIFIED BY '{model_connection.password}';")
    return model_connection, model_connection_name


def upload_language_container(conn, language_container):
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
    conn.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{alter_session}'")
    with open(container_path, "rb") as container_file:
        container_bucketfs_location.upload_fileobj_to_bucketfs(container_file, "ml.tar")


def create_input_table(conn):
    try:
        conn.execute("""
    DROP SCHEMA TARGET_SCHEMA CASCADE;
    """)
    except:
        pass
    conn.execute("""CREATE SCHEMA TARGET_SCHEMA;""")
    conn.execute("""
        CREATE OR REPLACE TABLE TEST.ABC(
            A FLOAT,
            B FLOAT,
            C FLOAT
        )
        """)
    for i in range(1, 1000):
        if i % 100 == 0:
            print(f"Insert {i}")
        values = ",".join([f"({j * 1.0 * i}, {j * 2.0 * i}, {j * 3.0 * i})" for j in range(1, 100)])
        conn.execute(f"INSERT INTO TEST.ABC VALUES {values}")
    print("COUNT", conn.execute("SELECT count(*) FROM TEST.ABC").fetchall())

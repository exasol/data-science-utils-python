import os
import subprocess
from pathlib import Path

import pyexasol
import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


def find_script(script_name: str):
    current_path = Path(".").absolute()
    script_path = None
    while (current_path != current_path.root):
        script_path = Path(current_path, script_name)
        if script_path.exists():
            break
        current_path = current_path.parent
    if script_path.exists():
        return script_path
    else:
        raise RuntimeError("Could not find build_language_container.sh")


@pytest.fixture(scope="session")
def language_container():
    script_dir = find_script("build_language_container.sh")
    completed_process = subprocess.run([script_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed_process.stdout.decode("UTF-8")
    print(output)
    completed_process.check_returncode()
    lines = output.splitlines()
    alter_session_selector = "ALTER SESSION SET SCRIPT_LANGUAGES='"
    alter_session = [line for line in lines if line.startswith(alter_session_selector)][0]
    alter_session = alter_session[len(alter_session_selector):-2]
    container_path_selector = "Cached container under "
    container_path = [line for line in lines if line.startswith(container_path_selector)][0]
    container_path = container_path[len(container_path_selector):]
    return {"container_path": container_path,
            "alter_session": alter_session}


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_data_science_utils_python.model_utils.udfs.train_udf import \
        TrainUDF

    udf = TrainUDF(exa)

    def run(ctx: UDFContext):
        udf.run(ctx)

    # @pytest.mark.skipif("CONTAINER_WITH_PROJECT" not in os.environ, reason="Container with project is not present")


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
    model_connection = Connection(address=f"http://localhost:6583/default/model;bfsdefault",
                                  user="w", password="write")
    db_connection = Connection(address=f"localhost:8888", user="sys", password="exasol")
    c = pyexasol.connect(dsn=db_connection.address, user=db_connection.user, password=db_connection.password)
    try:
        c.execute("""
    DROP SCHEMA TARGET_SCHEMA CASCADE;
    """)
    except:
        pass
    c.execute("""
CREATE SCHEMA TARGET_SCHEMA;
""")
    c.execute("""
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
        c.execute(
            f"INSERT INTO TEST.ABC VALUES {values}")

    print("COUNT", c.execute("SELECT count(*) FROM TEST.ABC").fetchall())
    exa = MockExaEnvironment(meta,
                             connections={
                                 "MODEL_CONNECTION": model_connection,
                                 "DB_CONNECTION": db_connection
                             })
    bucket_fs_factory = BucketFSFactory()
    model_bucketfs_location = \
        bucket_fs_factory.create_bucketfs_location(
            url=model_connection.address,
            user=model_connection.user,
            pwd=model_connection.password,
            base_path=None)

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
    print("alter_session", alter_session)
    print("container_path", container_path)
    c.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{alter_session}'")
    with open(container_path, "rb") as container_file:
        container_bucketfs_location.upload_fileobj_to_bucketfs(container_file, "ml.tar")
    model_connection_name = "MODEL_CONNECTION"
    try:
        c.execute(f"DROP CONNECTION {model_connection_name}")
    except:
        pass
    c.execute(
        f"CREATE CONNECTION {model_connection_name} TO 'http://localhost:6583/default/model;bfsdefault' USER '{model_connection.user}' IDENTIFIED BY '{model_connection.password}';")

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

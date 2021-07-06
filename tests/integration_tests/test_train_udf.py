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
    from exasol_data_science_utils_python.model_utils.udfs.train_udf import \
        TrainUDF

    udf = TrainUDF(exa)

    def run(ctx: UDFContext):
        udf.run(ctx)


def test_train_udf():
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
        ],
        output_type="EMIT",
        output_columns=[
            Column("output_model_path", str, "VARCHAR(2000000)"),
        ]
    )
    model_connection = Connection(address=f"http://localhost:6666/default/model;bfsdefault",
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
        A INTEGER,
        B VARCHAR(200000),
        C FLOAT
    )
    """)
    c.execute("INSERT INTO TEST.ABC VALUES (1,'A',1.0),(2,'B',2.0),(3,'C',3.0)")
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

    input_data = []
    input_data.append(
        (
            "MODEL_CONNECTION",
            "DB_CONNECTION",
            "TEST",
            "ABC",
            "A,B",
            "C",
            "TARGET_SCHEMA"
        )
    )
    result = list(executor.run([Group(input_data)], exa))

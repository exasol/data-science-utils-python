from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_data_science_utils_python.udf_utils.iterator_utils import ctx_iterator

    def run(ctx: UDFContext):
        iter = ctx_iterator(ctx, 10, lambda: ctx.reset())
        i = 0
        for b in iter:
            print(i)
            i = i + 1
            ctx.emit(b)


def test_partial_fit_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"), ],
        output_type="EMIT",
        output_columns=[Column("t1", int, "INTEGER"),
                        Column("t2", float, "FLOAT")]
    )
    exa = MockExaEnvironment(meta)
    input_data = [(i, (1.0 * i) / 105) for i in range(105)]
    result = executor.run([Group(input_data)], exa)
    assert len(result[0]) == 105

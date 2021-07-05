from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from sklearn.linear_model import SGDRegressor
    from sklearn.compose import ColumnTransformer
    from numpy.random import RandomState
    from sklearn.preprocessing import FunctionTransformer

    from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator

    def run(ctx: UDFContext):
        input_preprocessor = ColumnTransformer(transformers=[
            ("t2", FunctionTransformer(), ["t2"])
        ])
        output_preprocessor = ColumnTransformer(transformers=[
            ("t2", FunctionTransformer(), ["t2"])
        ])
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False, max_iter=100000,
                             tol=1e-10)
        df = ctx.get_dataframe(101)
        input_preprocessor.fit(df)
        output_preprocessor.fit(df)
        X = input_preprocessor.transform(df)
        y = output_preprocessor.transform(df)
        model.fit(X, y)
        iterator = ScoreIterator(
            input_preprocessor=input_preprocessor,
            output_preprocessor=output_preprocessor,
            model=model
        )
        score_sum, score_count = iterator.compute_score(ctx, 10)
        ctx.emit(score_sum, score_count)


def test_score_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"), ],
        output_type="EMIT",
        output_columns=[Column("SCORE_SUM", float, "FLOAD"),
                        Column("SCORE_COUNT", int, "INT"), ]
    )
    exa = MockExaEnvironment(meta)
    input_data = [(i, (1.0 * i) / 100) for i in range(100)]
    result = executor.run([Group(input_data)], exa)
    result_row = result[0].rows[0]
    assert result_row[1] == 100
    assert result_row[0] >= 99.9
    print(result_row[0] / result_row[1])

from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from sklearn.linear_model import SGDRegressor
    from sklearn.compose import ColumnTransformer
    from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
    from numpy.random import RandomState
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline

    def run(ctx: UDFContext):
        input_preprocessor = ColumnTransformer(transformers=[
            ("t2", FunctionTransformer(), ["t2"])
        ])
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False, max_iter=100000, tol=1e-10)
        pipeline = Pipeline([("p", input_preprocessor), ("m", model)])
        df = ctx.get_dataframe(101)
        pipeline.fit(df, df["t2"])
        iterator = PredictionIterator(
            input_preprocessor=input_preprocessor,
            model=model
        )
        iterator.predict(ctx, 10, lambda result: ctx.emit(result))


def test_prediction_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"), ],
        output_type="EMIT",
        output_columns=[Column("t1", int, "INTEGER"),
                        Column("t2", float, "FLOAT"),
                        Column("predicted_result", float, "FLOAT")]
    )
    exa = MockExaEnvironment(meta)
    input_data = [(i, (1.0 * i) / 100) for i in range(100)]
    result = executor.run([Group(input_data)], exa)
    result_rounded = [(t1, round(predicted_result, 2)) for t1, t2, predicted_result in result[0].rows]
    assert result_rounded == input_data

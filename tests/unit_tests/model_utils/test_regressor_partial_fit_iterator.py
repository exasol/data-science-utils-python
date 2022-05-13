from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor




def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from sklearn.linear_model import SGDRegressor
    from numpy.random import RandomState
    from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_identity_transformer import \
        SKLearnIdentityTransformer
    from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
        SKLearnPrefittedColumnTransformer
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
        ColumnSetPreprocessor
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
    from exasol_data_science_utils_python.model_utils.model_aggregator import combine_to_voting_regressor
    from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator

    from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator

    def run(ctx: UDFContext):
        input_preprocessor = SKLearnPrefittedColumnTransformer(
            transformer_mapping=[("t2", SKLearnIdentityTransformer())]
        )
        output_preprocessor = SKLearnPrefittedColumnTransformer(
            transformer_mapping=[("t2", SKLearnIdentityTransformer())]
        )
        table_preprocessor = TablePreprocessor(
            input_column_set_preprocessors=ColumnSetPreprocessor(
                column_transformer=input_preprocessor,
            ),
            target_column_set_preprocessors=ColumnSetPreprocessor(
                column_transformer=output_preprocessor,
            ),
        )

        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
        df = ctx.get_dataframe(101)
        iterator = RegressorPartialFitIterator(
            table_preprocessor=table_preprocessor,
            model=model
        )
        epochs = 10
        for i in range(epochs):
            iterator.train(ctx, batch_size=50, shuffle_buffer_size=100)
        combined_model = combine_to_voting_regressor([iterator.model, iterator.model])
        combined_iterator = ScoreIterator(
            table_preprocessor=iterator.table_preprocessor,
            model=combined_model
        )
        score_sum, score_count = combined_iterator.compute_score(ctx, batch_size=10)
        ctx.emit(score_sum, score_count)


def test_partial_fit_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"), ],
        output_type="EMITS",
        output_columns=[Column("SCORE_SUM", float, "FLOAD"),
                        Column("SCORE_COUNT", int, "INT"), ]
    )
    exa = MockExaEnvironment(meta)
    input_data = [(i, (1.0 * i) / 100) for i in range(100)]
    result = executor.run([Group(input_data)], exa)
    result_row = result[0].rows[0]
    assert result_row[1] == 100
    assert result_row[0] >= -5000.0
    print(result_row[0] / result_row[1])

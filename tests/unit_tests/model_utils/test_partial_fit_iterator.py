from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor




def udf_wrapper():
    from sklearn.linear_model import SGDClassifier
    from exasol_data_science_utils_python.model_utils.partial_fit_iterator import PartialFitIterator
    from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig

    def run(ctx):
        classifier = SGDClassifier()
        iterator_config = IteratorConfig(
            categorical_input_column_names=["t1"],
            numerical_input_column_names=["t2"],
            target_column_name="t3",
            input_column_category_counts=[100],
            target_classes=10,
        )
        iterator = PartialFitIterator(
            iterator_config=iterator_config,
            classifier=classifier
        )
        epochs = 5
        for i in range(epochs):
            iterator.train(ctx, batch_size=10, )
            score = iterator.compute_score(ctx, batch_size=10, )
            iterator.predict(ctx, 10, lambda result: ctx.emit(result))
            ctx.emit(None, score, None, None)
        voting_classifier_iterator = PartialFitIterator.combine_to_voting_classifier([iterator, iterator])
        score = voting_classifier_iterator.compute_score(ctx, batch_size=10, )
        voting_classifier_iterator.predict(ctx, 10, lambda result: ctx.emit(result))
        ctx.emit(None, score, None, None)


def test_partial_fit_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"),
                       Column("t3", int, "INTEGER")],
        output_type="EMIT",
        output_columns=[Column("t1", int, "INTEGER"),
                        Column("t2", float, "FLOAT"),
                        Column("t3", int, "INTEGER"),
                        Column("predicted_result", int, "INTEGER")]
    )
    exa = MockExaEnvironment(meta)
    classes = 10
    result = executor.run([Group([(i, (1.0 * i) / 100, i % classes) for i in range(100)])], exa)
    print(result)
    assert result[0].rows[-1][1] == 1.0

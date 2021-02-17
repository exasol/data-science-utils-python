from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor


def udf_wrapper():
    from sklearn.linear_model import SGDClassifier
    from exasol_data_science_utils_python.model_utils.partial_fit_iterator import PartialFitIterator

    def run(ctx):
        classifier = SGDClassifier()
        iterator = PartialFitIterator(
            ctx,
            batch_size=10,
            categorical_input_column_names=["t1"],
            numerical_input_column_names=["t2"],
            target_column_name="t3",
            input_column_category_counts=[100],
            target_classes=100,
            classifier=classifier
        )
        iterator.train()
        ctx.emit(str(classifier.intercept_))


def test_partial_fit_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"),
                       Column("t3", int, "INTEGER")],
        output_type="EMIT",
        output_columns=[Column("t3", str, "VARCHAR(20000)")]
    )
    exa = MockExaEnvironment(meta)
    result = executor.run([Group([(i, 1.0 * i, i) for i in range(100)])], exa)
    print(result)

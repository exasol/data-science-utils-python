from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from sklearn.ensemble import RandomForestClassifier

from exasol_data_science_utils_python.model_utils.persistence import load_from_base64_string


def udf_wrapper():
    from sklearn.ensemble import RandomForestClassifier
    from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
        SKLearnPrefittedColumnTransformer
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
        ColumnSetPreprocessor
    from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
    from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_identity_transformer import \
        SKLearnIdentityTransformer

    from exasol_data_science_utils_python.model_utils.random_forest_iterator import RandomForestIterator

    def run(ctx):
        model = RandomForestClassifier(random_state=0)
        input_preprocessor = SKLearnPrefittedColumnTransformer(transformer_mapping=[
            ("t1", SKLearnIdentityTransformer()),
            ("t2", SKLearnIdentityTransformer()),
        ])
        output_preprocessor = SKLearnPrefittedColumnTransformer(transformer_mapping=[
            ("t3", SKLearnIdentityTransformer()),
        ])
        table_preprocessor = TablePreprocessor(
            input_column_set_preprocessors=ColumnSetPreprocessor(
                column_transformer=input_preprocessor,
            ),
            target_column_set_preprocessors=ColumnSetPreprocessor(
                column_transformer=output_preprocessor,
            ),
        )

        iterator = RandomForestIterator(
            table_preprocessor=table_preprocessor,
            target_classes=10,
            model=model
        )
        iterator.train(ctx, batch_size=9)


def test_random_forest_iterator():
    executor = UDFMockExecutor()
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[Column("t1", int, "INTEGER"),
                       Column("t2", float, "FLOAT"),
                       Column("t3", int, "INTEGER")],
        output_type="EMIT",
        output_columns=[Column("serialized_random_forest", str, "VARCHAR(2000000)")]
    )
    exa = MockExaEnvironment(meta)
    classes = 10
    result = executor.run([Group([(i, (1.0 * i) / 100, i % classes) for i in range(100)])], exa)
    assert len(result[0].rows) == 12
    estimators = []
    for r in result[0].rows:
        value = load_from_base64_string(r[0])
        assert isinstance(value, RandomForestClassifier)
        estimators.append(value)

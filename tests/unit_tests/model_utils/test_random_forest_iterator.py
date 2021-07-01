import pandas as pd
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig
from exasol_data_science_utils_python.model_utils.persistence import load_from_base64_string
from exasol_data_science_utils_python.model_utils.random_forest_iterator import RandomForestIterator


def udf_wrapper():
    from sklearn.ensemble import RandomForestClassifier

    from exasol_data_science_utils_python.model_utils.random_forest_iterator import RandomForestIterator
    from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig

    def run(ctx):
        classifier = RandomForestClassifier(random_state=0)
        iterator_config = IteratorConfig(
            categorical_input_column_names=["t1"],
            numerical_input_column_names=["t2"],
            target_column_name="t3",
            input_column_category_counts=[100],
            target_classes=10,
        )
        iterator = RandomForestIterator(
            iterator_config=iterator_config,
            classifier=classifier
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
    assert len(result[0].rows) == 11
    estimators = []
    for r in result[0].rows:
        value = load_from_base64_string(r[0])
        assert isinstance(value, RandomForestIterator)
        estimators.append(value)
    combinedIterator = RandomForestIterator.combine_to_random_forrest(estimators)
    print(combinedIterator)
    batch = pd.DataFrame.from_records([(i, (1.0 * i) / 100, i % classes) for i in range(100)],
                                      columns=["t1", "t2", "t3"])
    result = combinedIterator._predict_batch(batch)
    score = combinedIterator._compute_score_batch(batch)
    print(result)
    print(score)
    assert score[1] > 85.0

from sklearn.linear_model import SGDClassifier

from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig
from exasol_data_science_utils_python.model_utils.partial_fit_iterator import PartialFitIterator


def test():
    classifier = SGDClassifier()
    iterator_config = IteratorConfig(
        categorical_input_column_names=["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"],
        numerical_input_column_names=["age", "fnlwgt", "education_num", "capitalgain", "capitalloss", "hoursperweek"],
        target_column_name="class",
        input_column_category_counts=[8, 16,  7, 14,  6,  5,  2, 41],
        target_classes=2,
    )
    iterator = PartialFitIterator(
        iterator_config=iterator_config,
        classifier=classifier
    )
from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig
from exasol_data_science_utils_python.model_utils.model_aggregator import combine_random_forrest_classifier
from exasol_data_science_utils_python.model_utils.persistence import dump_to_base64_string
from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class RandomForestIterator(PredictionIterator):

    def __init__(self,
                 iterator_config: IteratorConfig,
                 classifier: RandomForestClassifier):
        super().__init__(iterator_config, classifier)
        getattr(classifier, "fit")
        if not isinstance(classifier, RandomForestClassifier):
            raise Exception("Not a random forrest classifier")

    def _train_batch(self, df: pd.DataFrame):
        input_columns, target_column = self._preprocess_batch(df)
        if len(target_column.unique()) != self.iterator_config.target_classes:
            # We need to append missing classes with dummy inputs,
            # because otherwise we can't merge the RandomForestClassifier later
            input_columns, target_column = self.append_missing_classes(input_columns, target_column)

        cloned_classifier = sklearn.base.clone(self.classifier)  # type: RandomForestClassifier
        cloned_classifier.fit(input_columns.values, target_column.values)
        random_forest_iterator = RandomForestIterator(
            self.iterator_config,
            cloned_classifier)
        b64_string = dump_to_base64_string(random_forest_iterator)
        return b64_string

    def append_missing_classes(self, input_columns, target_column):
        missing_classes = set(range(self.iterator_config.target_classes)).difference(set(target_column.to_list()))
        if len(missing_classes) > 1 and len(missing_classes) / self.iterator_config.target_classes > 0.1:
            raise Exception(
                f"to many classes are missing the current batch expected {self.iterator_config.target_classes} "
                f"got {len(missing_classes)}, either increase the batch_size or reshuffle the data")
        missing_target_rows = pd.Series(list(missing_classes))
        target_column = target_column.append(missing_target_rows)
        missing_input_rows = pd.DataFrame(
            np.zeros(shape=(len(missing_classes), input_columns.shape[1]), dtype=np.float64),
            columns=input_columns.columns
        )
        input_columns = input_columns.append(missing_input_rows)
        return input_columns, target_column

    def train(self, ctx, batch_size: int):
        iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._train_batch(df),
            lambda: None,
            lambda state, result: ctx.emit(result),
            lambda: ctx.reset()
        )

    @classmethod
    def combine_to_random_forrest(cls, rf_estimators: List["RandomForestIterator"]) -> "RandomForestIterator":
        for estimator in rf_estimators:
            if not estimator.is_compatible(rf_estimators[0]):
                raise Exception("Estimators are not compatible")
        estimators_ = [estimator.classifier for estimator in rf_estimators]
        voting_classifiier = combine_random_forrest_classifier(estimators_)
        return RandomForestIterator(
            iterator_config=rf_estimators[0].iterator_config,
            classifier=voting_classifiier
        )

from typing import Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from exasol_data_science_utils_python.model_utils.persistence import dump_to_base64_string
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class RandomForestIterator(ScoreIterator):
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 target_classes: int,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(input_preprocessor, output_preprocessor, model)
        self.target_classes = target_classes
        self.output_preprocessor = output_preprocessor
        self.model = model
        getattr(model, "fit")

    def _train_batch(self, df: pd.DataFrame):
        input_columns = self.input_preprocessor.transform(df)
        target_columns = self.output_preprocessor.transform(df)
        if target_columns.shape[1] > 1:
            raise Exception("target columns have more than 1 column")
        if len(np.unique(target_columns)) != self.target_classes:
            # We need to append missing classes with dummy inputs,
            # because otherwise we can't merge the RandomForestClassifier later
            input_columns, target_columns = self.append_missing_classes(input_columns, target_columns)

        cloned_classifier = sklearn.base.clone(self.model)  # type: RandomForestClassifier
        cloned_classifier.fit(input_columns, target_columns)
        b64_string = dump_to_base64_string(cloned_classifier)
        return b64_string

    def append_missing_classes(self, input_columns: np.array, target_columns: np.array):
        classes_in_target_columns = set(np.reshape(target_columns, [-1]).tolist())
        missing_classes = set(range(self.target_classes)).difference(classes_in_target_columns)
        # if len(missing_classes) > 1 and len(missing_classes) / self.target_classes > 0.05:
        #     raise Exception(
        #         f"to many classes are missing the current batch expected {self.target_classes} "
        #         f"got {len(missing_classes)}, either increase the batch_size or reshuffle the data")
        missing_classes_array = np.array([list(missing_classes)]).reshape((-1, 1))
        target_columns = np.vstack([target_columns, missing_classes_array])
        missing_input_rows = np.zeros(shape=(len(missing_classes), input_columns.shape[1]), dtype=np.float64)
        input_columns = np.vstack([input_columns, missing_input_rows])
        return input_columns, target_columns

    def train(self, ctx, batch_size: int):
        iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._train_batch(df),
            lambda: None,
            lambda state, result: ctx.emit(result),
            lambda: ctx.reset()
        )
    #
    # @classmethod
    # def combine_to_random_forrest(cls, rf_estimators: List["RandomForestIterator"]) -> "RandomForestIterator":
    #     for estimator in rf_estimators:
    #         if not estimator.is_compatible(rf_estimators[0]):
    #             raise Exception("Estimators are not compatible")
    #     estimators_ = [estimator.model for estimator in rf_estimators]
    #     voting_classifiier = combine_random_forrest_classifier(estimators_)
    #     return RandomForestIterator(
    #         iterator_config=rf_estimators[0].iterator_config,
    #         classifier=voting_classifiier
    #     )

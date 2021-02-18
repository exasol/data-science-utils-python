from typing import List, Any

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig
from exasol_data_science_utils_python.model_utils.model_aggregator import combine_to_voting_classifier
from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset

# See more at https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=online#incremental-learning

class PartialFitIterator(PredictionIterator):

    def __init__(self,
                 iterator_config: IteratorConfig,
                 classifier: Any):
        super().__init__(iterator_config, classifier)
        getattr(classifier, "partial_fit")

    def _train_batch(self, df: pd.DataFrame):
        input_columns, target_column = self._preprocess_batch(df)
        target_classes = np.arange(self.iterator_config.target_classes)
        self.classifier.partial_fit(input_columns.values, target_column.values, classes=target_classes)

    def train(self, ctx, batch_size: int):
        iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._train_batch(df),
            lambda: None,
            lambda state, result: None,
            lambda: ctx.reset()
        )

    @classmethod
    def combine_to_voting_classifier(clazz, partial_fit_estimators: List["PartialFitIterator"], **kwargs):
        for estimator in partial_fit_estimators:
            if not estimator.is_compatible(partial_fit_estimators[0]):
                raise Exception("Estimators are not compatible")
        estimators_ = [estimator.classifier for estimator in partial_fit_estimators]
        voting_classifiier = combine_to_voting_classifier(estimators_,
                                                          partial_fit_estimators[0].iterator_config.target_classes,
                                                          **kwargs)
        return PredictionIterator(
            iterator_config=partial_fit_estimators[0].iterator_config,
            classifier=voting_classifiier
        )

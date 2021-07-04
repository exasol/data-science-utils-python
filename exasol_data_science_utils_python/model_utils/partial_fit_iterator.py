from typing import Union

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.model_utils.reservoir_shuffle import ReservoirShuffle
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import ctx_iterator


# See more at https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=online#incremental-learning

class PartialFitIterator(ScoreIterator):
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(input_preprocessor, output_preprocessor, model)
        self.output_preprocessor = output_preprocessor
        self.model = model
        getattr(model, "partial_fit")

    def _train_batch(self, df: pd.DataFrame):
        input_columns = self.input_preprocessor.transform(df)
        output_columns = self.output_preprocessor.transform(df)
        self.model.partial_fit(input_columns, output_columns)

    def train(self, ctx, batch_size: int, shuffle_buffer_size: int):
        input_iter = ctx_iterator(ctx, batch_size, lambda: ctx.reset())
        shuffle_iter = ReservoirShuffle(input_iter, shuffle_buffer_size, batch_size).shuffle()
        for df in shuffle_iter:
            self._train_batch(df)
    #
    # @classmethod
    # def combine_to_voting_classifier(clazz, partial_fit_estimators: List["PartialFitIterator"], **kwargs):
    #     for estimator in partial_fit_estimators:
    #         if not estimator.is_compatible(partial_fit_estimators[0]):
    #             raise Exception("Estimators are not compatible")
    #     estimators_ = [estimator.model for estimator in partial_fit_estimators]
    #     voting_classifiier = combine_to_voting_classifier(estimators_,
    #                                                       partial_fit_estimators[0].iterator_config.target_classes,
    #                                                       **kwargs)
    #     return PredictionIterator(
    #         iterator_config=partial_fit_estimators[0].iterator_config,
    #         classifier=voting_classifiier
    #     )

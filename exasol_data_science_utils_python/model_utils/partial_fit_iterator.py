from abc import ABC, abstractmethod
from typing import Union, List

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.model_utils.model_aggregator import combine_to_voting_regressor
from exasol_data_science_utils_python.model_utils.reservoir_shuffle import ReservoirShuffle
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import ctx_iterator


# See more at https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=online#incremental-learning

class PartialFitIterator(ScoreIterator, ABC):
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(input_preprocessor, output_preprocessor, model)
        self.model = model
        getattr(model, "partial_fit")

    def _train_batch(self, df: pd.DataFrame):
        input_df = df[self.input_preprocessor_columns]
        output_df = df[self.output_preprocessor_columns]
        input_columns = self.input_preprocessor.transform(input_df)
        output_columns = self.output_preprocessor.transform(output_df)
        self.run_partial_fit(input_columns, output_columns)

    @abstractmethod
    def run_partial_fit(self, input_columns, output_columns):
        pass

    def train(self, ctx, batch_size: int, shuffle_buffer_size: int):
        input_iter = ctx_iterator(ctx, batch_size, lambda: ctx.reset())
        shuffle_iter = ReservoirShuffle(input_iter, shuffle_buffer_size, batch_size).shuffle()
        for df in shuffle_iter:
            self._train_batch(df)


class RegressorPartialFitIterator(PartialFitIterator):

    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 model: Union[RegressorMixin]):
        super().__init__(input_preprocessor, output_preprocessor, model)

    def run_partial_fit(self, input_columns, output_columns):
        self.model.partial_fit(input_columns, output_columns)

    @classmethod
    def combine_to_voting_regressor(clazz, partial_fit_estimators: List["RegressorPartialFitIterator"],
                                    **kwargs) -> ScoreIterator:
        """
        Combines multiple PartialFitIterators into one ScoreIterator using combine_to_voting_regressor
        Note: We can't compare Models or ColumnTransformers, as such we can't guarantee
              that the PartialFitIterators are compatible
        :param partial_fit_estimators:
        :param kwargs: kwargs for combine_to_voting_classifier
        :return: ScoreIterator with the input and output preprocessor of the first PartialFitIterator
                 and the combined models of PartialFitIterators
        """
        estimators_ = [estimator.model for estimator in partial_fit_estimators]
        voting_regessor = combine_to_voting_regressor(estimators_,
                                                         **kwargs)
        return ScoreIterator(
            model=voting_regessor,
            input_preprocessor=partial_fit_estimators[0].input_preprocessor,
            output_preprocessor=partial_fit_estimators[0].output_preprocessor
        )


class ClassifierPartialFitIterator(PartialFitIterator):
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 model: Union[ClassifierMixin],
                 classes: Union[List[int], List[str]]):
        super().__init__(input_preprocessor, output_preprocessor, model)
        self.classes = classes

    def run_partial_fit(self, input_columns, output_columns):
        self.model.partial_fit(input_columns, output_columns, classes=self.classes)

    # TODO test combine_to_voting_classifier with different class lists
    # @classmethod
    # def combine_to_voting_classifier(clazz, partial_fit_estimators: List["ClassifierPartialFitIterator"],
    #                                  **kwargs) -> ScoreIterator:
    #     """
    #     Combines multiple PartialFitIterators into one ScoreIterator using combine_to_voting_classifier
    #     Note: We can't compare Models or ColumnTransformers, as such we can't guarantee
    #           that the PartialFitIterators are compatible
    #     :param partial_fit_estimators:
    #     :param kwargs: kwargs for combine_to_voting_classifier
    #     :return: ScoreIterator with the input and output preprocessor of the first PartialFitIterator
    #              and the combined models of PartialFitIterators
    #     """
    #     estimators_ = [estimator.model for estimator in partial_fit_estimators]
    #     voting_classifiier = combine_to_voting_classifier(estimators_,
    #                                                       partial_fit_estimators[0].classes,
    #                                                       **kwargs)
    #     return ScoreIterator(
    #         model=voting_classifiier,
    #         input_preprocessor=partial_fit_estimators[0].input_preprocessor,
    #         output_preprocessor=partial_fit_estimators[0].output_preprocessor
    #     )

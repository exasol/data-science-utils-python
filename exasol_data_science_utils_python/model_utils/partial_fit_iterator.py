from abc import ABC, abstractmethod
from typing import Union, List

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.model_utils.reservoir_shuffle import ReservoirShuffle
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.udf_utils.iterator_utils import ctx_iterator


# See more at https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=online#incremental-learning

class PartialFitIterator(ScoreIterator, ABC):
    def __init__(self,
                 table_preprocessor:TablePreprocessor,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(table_preprocessor, model)
        self.model = model
        getattr(model, "partial_fit")

    def _train_batch(self, df: pd.DataFrame):
        input_columns = self._get_input_columns(df)
        output_columns = self._get_output_columns(df)
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
                 table_preprocessor:TablePreprocessor,
                 model: Union[RegressorMixin]):
        super().__init__(table_preprocessor, model)

    def run_partial_fit(self, input_columns, output_columns):
        self.model.partial_fit(input_columns, output_columns)


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
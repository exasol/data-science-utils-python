from typing import Union

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class ScoreIterator(PredictionIterator):
    def __init__(self,
                 table_preprocessor: TablePreprocessor,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(table_preprocessor, model)
        getattr(model, "score")
        self.model = model

    def _compute_score_batch(self, df: pd.DataFrame):
        input_columns = self._get_input_columns(df)
        output_columns = self._get_output_columns(df)
        score = self.model.score(input_columns, output_columns)
        return len(df), len(df) * score

    def _get_output_columns(self, df):
        output_columns = self.table_preprocessor.target_column_set_preprocessors.column_transformer.transform(df)
        return output_columns

    def compute_score(self, ctx, batch_size: int):
        final_state = iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._compute_score_batch(df),
            lambda: {"count": 0, "score_sum": 0},
            lambda state, result: {"count": state["count"] + result[0], "score_sum": state["score_sum"] + result[1]},
            lambda: ctx.reset()
        )
        return final_state["score_sum"], final_state["count"]

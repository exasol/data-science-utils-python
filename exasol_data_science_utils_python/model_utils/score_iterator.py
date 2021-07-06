from typing import Union

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class ScoreIterator(PredictionIterator):
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 output_preprocessor: ColumnTransformer,
                 model: Union[ClassifierMixin, RegressorMixin]):
        super().__init__(input_preprocessor, model)
        self.output_preprocessor = output_preprocessor
        self.output_preprocessor_columns = self._get_columns_from_column_transformer(self.output_preprocessor)
        getattr(model, "score")
        self.model = model

    def _compute_score_batch(self, df: pd.DataFrame):
        input_df = df[self.input_preprocessor_columns]
        output_df = df[self.output_preprocessor_columns]
        input_columns = self.input_preprocessor.transform(input_df)
        output_columns = self.output_preprocessor.transform(output_df)
        score = self.model.score(input_columns, output_columns)
        return len(df), len(df) * score

    def compute_score(self, ctx, batch_size: int):
        final_state = iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._compute_score_batch(df),
            lambda: {"count": 0, "score_sum": 0},
            lambda state, result: {"count": state["count"] + result[0], "score_sum": state["score_sum"] + result[1]},
            lambda: ctx.reset()
        )
        return final_state["score_sum"], final_state["count"]

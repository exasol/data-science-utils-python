from typing import Callable, Union

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class PredictionIterator:
    def __init__(self,
                 input_preprocessor: ColumnTransformer,
                 model: Union[ClassifierMixin, RegressorMixin]
                 ):
        self.input_preprocessor = input_preprocessor
        getattr(model, "predict")
        self.model = model

    def _predict_batch(self, df: pd.DataFrame):
        input_columns = self.input_preprocessor.transform(df)
        result = self.model.predict(input_columns)
        df["predicted_result"] = result
        return df

    def predict(self, ctx, batch_size: int, result_callback: Callable[[pd.DataFrame], None]):
        iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._predict_batch(df),
            lambda: None,
            lambda state, result: result_callback(result),
            lambda: ctx.reset()
        )

import itertools
from typing import Callable, Union, List

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
        self.input_preprocessor_columns = self._get_columns_from_column_transformer(self.input_preprocessor)
        getattr(model, "predict")
        self.model = model

    def _get_columns_from_column_transformer(self, column_tansformer: ColumnTransformer) -> List[str]:
        column_lists = [columns for name, transformer, columns in column_tansformer.transformers]
        columns = list(itertools.chain.from_iterable(column_lists))
        return columns

    def _predict_batch(self, df: pd.DataFrame):
        input_df = df[self.input_preprocessor_columns]
        input_columns = self.input_preprocessor.transform(input_df)
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

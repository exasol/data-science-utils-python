from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig
from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class PredictionIterator:
    def __init__(self,
                 iterator_config: IteratorConfig,
                 classifier: Any
                 ):
        self.iterator_config = iterator_config
        getattr(classifier, "score")
        getattr(classifier, "predict")
        self.classifier = classifier
        self.input_column_one_hot_encoder = self._create_one_hot_encoder()

    def _create_one_hot_encoder(self):
        categories = [np.arange(i) for i in self.iterator_config.input_column_category_counts]
        # TODO in older versions of scikit learn it was n_values
        one_hot_encoder = OneHotEncoder(categories=categories, sparse=False)
        data = np.ones(shape=(len(self.iterator_config.input_column_category_counts), 1))
        one_hot_encoder.fit(data)
        return one_hot_encoder

    def _encode_categorical_columns(self, categorical_columns: pd.DataFrame):
        transformed_columns = pd.DataFrame(
            self.input_column_one_hot_encoder.transform(categorical_columns.values))
        return transformed_columns

    def _preprocess_batch(self, df: pd.DataFrame, prdict: bool = False):
        encoded_input_categorical_columns = \
            self._encode_categorical_columns(df[self.iterator_config.categorical_input_column_names])
        numerical_input_columns = df[self.iterator_config.numerical_input_column_names].astype(float)
        input_columns = pd.concat([encoded_input_categorical_columns,
                                   numerical_input_columns], axis=1)
        if not prdict:
            target_column = df[self.iterator_config.target_column_name].astype(int)
        else:
            target_column = None
        return input_columns, target_column

    def _compute_score_batch(self, df: pd.DataFrame):
        input_columns, target_column = self._preprocess_batch(df)
        score = self.classifier.score(input_columns.values, target_column.values)
        return len(df), len(df) * score

    def compute_score(self, ctx, batch_size: int):
        final_state = iterate_trough_dataset(
            ctx, batch_size,
            lambda df: self._compute_score_batch(df),
            lambda: {"count": 0, "score_sum": 0},
            lambda state, result: {"count": state["count"] + result[0], "score_sum": state["score_sum"] + result[1]},
            lambda: ctx.reset()
        )
        score = final_state["score_sum"] / final_state["count"]
        return score

    def _predict_batch(self, df: pd.DataFrame):
        input_columns, _ = self._preprocess_batch(df, prdict=True)
        result = self.classifier.predict(input_columns.values)
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

    def is_compatible(self, estimator: "PredictionIterator") -> bool:
        return self.iterator_config.is_compatible(estimator.iterator_config)

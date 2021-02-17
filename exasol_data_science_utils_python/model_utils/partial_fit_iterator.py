from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from exasol_data_science_utils_python.udf_utils.iterator_utils import iterate_trough_dataset


class PartialFitIterator:

    def __init__(self, ctx,
                 batch_size: int,
                 categorical_input_column_names: List[str],
                 numerical_input_column_names: List[str],
                 target_column_name: str,
                 input_column_category_counts: List[int],
                 target_classes: int,
                 classifier: Any
                 ):
        self.ctx = ctx
        self.batch_size = batch_size
        self.categorical_input_column_names = categorical_input_column_names
        self.numerical_input_column_names = numerical_input_column_names
        self.target_column_name = target_column_name
        self.input_column_category_counts = input_column_category_counts
        self.target_classes = target_classes
        getattr(classifier, "partial_fit")
        self.classifier = classifier
        self.input_column_one_hot_encoder = self._create_one_hot_encoder()

    def set_ctx(self, ctx):
        self.ctx = ctx

    def _reset(self):
        self.ctx.reset()

    def _create_one_hot_encoder(self):
        categories = [np.arange(i) for i in self.input_column_category_counts]
        # TODO in older versions of scikit learn it was n_values
        one_hot_encoder = OneHotEncoder(categories=categories, sparse=False)
        self._reset()
        data = self.ctx.get_dataframe(2)[self.categorical_input_column_names].astype(int).values
        one_hot_encoder.fit(data)
        return one_hot_encoder

    def _encode_categorical_columns(self, categorical_columns: pd.DataFrame):
        transformed_columns = pd.DataFrame(
            self.input_column_one_hot_encoder.transform(categorical_columns.values))
        return transformed_columns

    def _preprocess_batch(self, df):
        encoded_input_categorical_columns = \
            self._encode_categorical_columns(df[self.categorical_input_column_names])
        numerical_input_columns = df[self.numerical_input_column_names].astype(float)
        input_columns = pd.concat([encoded_input_categorical_columns,
                                   numerical_input_columns], axis=1)
        target_column = df[self.target_column_name].astype(int)
        return input_columns, target_column

    def _train_batch(self, df):
        input_columns, target_column = self._preprocess_batch(df)
        target_classes = np.arange(self.target_classes)
        self.classifier.partial_fit(input_columns.values, target_column.values, classes=target_classes)

    def train(self):
        iterate_trough_dataset(
            self.ctx, self.batch_size,
            lambda df: self._train_batch(df),
            lambda: None,
            lambda state, result: None,
            lambda: self._reset()
        )

    def compute_score(self):
        final_state = iterate_trough_dataset(
            self.ctx, self.batch_size,
            lambda df: self.compute_score_batch(df),
            lambda: {"count": 0, "score_sum": 0},
            lambda state, result: {"count": state["count"] + result[0], "score_sum": state["score_sum"] + result[1]},
            lambda: self._reset()
        )
        score = final_state["score_sum"] / final_state["count"]
        return score

    def compute_score_batch(self, df):
        input_columns, target_column = self._preprocess_batch(df)
        score = self.classifier.score(input_columns.values, target_column.values)
        return len(df), len(df) * score

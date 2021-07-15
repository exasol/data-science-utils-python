from typing import Tuple, List

import numpy as np

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import \
    SKLearnPrefittedTransformer


class SKLearnPrefittedColumnTransformer(SKLearnPrefittedTransformer):
    def __init__(self, transformer_mapping: List[Tuple[str, SKLearnPrefittedTransformer]]):
        self._transformer_mapping = transformer_mapping

    def transform(self, X, y=None):
        transformed_columns = \
            [transformer.transform(X[column])
             for column, transformer in self._transformer_mapping]
        return np.hstack(transformed_columns)

from typing import Tuple, List, Union

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import \
    SKLearnPrefittedTransformer


class SKLearnPrefittedColumnTransformer(SKLearnPrefittedTransformer):
    def __init__(self, transformer_mapping: List[Tuple[str, SKLearnPrefittedTransformer]]) -> object:
        self._transformer_mapping = transformer_mapping

    def align_column(self,transformed_column:Union[pd.DataFrame,pd.Series,np.ndarray]):
        if isinstance(transformed_column,pd.DataFrame):
            return transformed_column.values
        elif isinstance(transformed_column,pd.Series):
            return transformed_column.values.reshape((-1,1))
        elif isinstance(transformed_column,np.ndarray):
            if len(transformed_column.shape)==1:
                return transformed_column.reshape((-1,1))
            else:
                return transformed_column
        else:
            raise TypeError(f"Type of transformed_column {type(transformed_column)} not supported")

    def transform(self, X, y=None):
        transformed_columns = \
            [transformer.transform(X[column])
             for column, transformer in self._transformer_mapping]
        aligned_transformed_columns = [self.align_column(transformed_column) for transformed_column in transformed_columns]
        hstack = np.hstack(aligned_transformed_columns)
        return hstack

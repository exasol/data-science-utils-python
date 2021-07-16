import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import \
    SKLearnPrefittedTransformer


class SKLearnPrefittedOneHotTransformer(SKLearnPrefittedTransformer):
    def __init__(self, dictionary: pd.DataFrame):
        self.dictionary = dictionary.set_index("VALUE", drop=False)
        columns_set = set(self.dictionary.columns.to_list())
        expected_column_set = {"ID", "VALUE"}
        if columns_set != expected_column_set:
            raise Exception(f"Columns of dictionary {columns_set} don't fit {expected_column_set}")

    def transform(self, X, y=None):
        if not (len(X.shape) == 1 or len(X.shape) == 2 and X.shape == 1):
            raise ValueError(f"X {X} can't be interpreted as column vector")
        reshape_X = X.values.reshape((-1,))
        setdiff_X_dictionary = np.setdiff1d(reshape_X, self.dictionary.index.values)
        if len(setdiff_X_dictionary) > 0:
            raise ValueError(
                f"The following values {setdiff_X_dictionary} in X are not in the dictionary")
        dictionary_size = len(self.dictionary)
        X_size = len(reshape_X)
        result_X = np.zeros(shape=(X_size, dictionary_size))
        for row, value in enumerate(reshape_X):
            result_X[row, self.dictionary.at[value, "ID"]] = 1
        result_X = result_X[:, 1:]
        return result_X

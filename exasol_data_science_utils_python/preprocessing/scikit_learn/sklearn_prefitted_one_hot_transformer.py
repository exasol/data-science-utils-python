import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import SKLearnPrefittedTransformer


class SKLearnPrefittedOneHotTransformer(SKLearnPrefittedTransformer):
    def __init__(self, dictionary: pd.DataFrame):
        self._dictionary = dictionary
        columns_set = set(self._dictionary.columns.to_list())
        expected_column_set = {"ID", "VALUE"}
        if columns_set != expected_column_set:
            raise Exception(f"Columns of dictionary {columns_set} don't fit {expected_column_set}")
        self._contrast_matrix = None

    def __getstate__(self):
        copy = dict(self.__dict__)
        copy["_contrast_matrix"] = None
        return copy

    def transform(self, X, y=None):
        if self._contrast_matrix is None:
            self._contrast_matrix = self.create_contrast_matrix()

        reshape_X = X.values.reshape((-1,))
        X_isin_index = self._dictionary.index.isin(reshape_X)
        if all(X_isin_index):
            raise ValueError(
                f"The following values {reshape_X[np.logical_not(X_isin_index)]} in X are not in the dictionary")
        X_ = self._contrast_matrix.loc[reshape_X].values
        return X_

    def create_contrast_matrix(self):
        dictionary_size = len(self._dictionary)
        contrast_matrix = np.zeros(shape=(dictionary_size, dictionary_size))
        for row, column in enumerate(self._dictionary["ID"]):
            contrast_matrix[row, column] = 1
        contrast_matrix = contrast_matrix[:,1:]
        contrast_matrix_df = pd.DataFrame(data=contrast_matrix, index=self._dictionary["VALUE"])
        return contrast_matrix_df

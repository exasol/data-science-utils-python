import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import SKLearnPrefittedTransformer


class SKLearnPrefittedContrastMatrixTransformer(SKLearnPrefittedTransformer):
    def __init__(self, contrast_matrix: pd.DataFrame):
        self.contrast_matrix = contrast_matrix

    def transform(self, X, y=None):
        reshape_X = X.values.reshape((-1,))
        X_ = self.contrast_matrix.loc[reshape_X].values
        return X_

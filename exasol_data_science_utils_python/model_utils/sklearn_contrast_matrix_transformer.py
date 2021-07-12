import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnContrastMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contrast_matrix: pd.DataFrame):
        self.contrast_matrix = contrast_matrix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        reshape_X = X.values.reshape((-1,))
        X_ = self.contrast_matrix.loc[reshape_X].values
        return X_

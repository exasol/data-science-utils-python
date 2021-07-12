from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnMinMaxScalar(BaseEstimator, TransformerMixin):
    def __init__(self, min_value: float, range_value: float):
        self.range_value = range_value
        self.min_value = min_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_std = (X - self.min_value) / self.range_value
        return X_std

from sklearn.base import BaseEstimator, TransformerMixin

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import SKLearnPrefittedTransformer


class SKLearnIdentityTransformer(SKLearnPrefittedTransformer):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return X

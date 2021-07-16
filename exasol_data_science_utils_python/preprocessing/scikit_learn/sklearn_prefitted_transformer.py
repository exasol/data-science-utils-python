from abc import abstractmethod, ABC

from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnPrefittedTransformer(BaseEstimator, TransformerMixin, ABC):
    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X, y=None):
        pass
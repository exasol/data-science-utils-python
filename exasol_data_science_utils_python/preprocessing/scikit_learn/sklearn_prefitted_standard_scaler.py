from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import \
    SKLearnPrefittedTransformer


class SKLearnPrefittedStandardScaler(SKLearnPrefittedTransformer):
    def __init__(self, avg_value: float, stddev_value: float):
        self._stddev_value = stddev_value
        if self._stddev_value == 0:
            self._effective_stddev_value = 1
        else:
            self._effective_stddev_value = self._stddev_value
        self._avg_value = avg_value

    def transform(self, X, y=None):
        X_minus_avg = (X - self._avg_value)
        X_std = X_minus_avg / self._effective_stddev_value
        return X_std

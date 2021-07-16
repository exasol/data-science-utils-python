from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_transformer import SKLearnPrefittedTransformer


class SKLearnPrefittedMinMaxScaler(SKLearnPrefittedTransformer):
    def __init__(self, min_value: float, range_value: float):
        self.range_value = range_value
        self.min_value = min_value

    def transform(self, X, y=None):
        X_zeroed = (X - self.min_value)
        X_std = X_zeroed / self.range_value
        return X_std

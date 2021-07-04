import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array


class Converter:

    def convert(self, *args, **kwargs):
        print(args[0].shape)
        rand = np.random.rand(args[0].shape[0], 10)
        print(rand.shape)
        return rand


def test_run():
    np.random.seed(0)

    # Load data from https://www.openml.org/d/40945
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    categorical_features = ['embarked', 'sex', 'pclass']
    converter = Converter()
    categorical_transformer = FunctionTransformer(func=converter.convert)

    categorical_preprocessor = ColumnTransformer(
        transformers=[
            (f, categorical_transformer, [f]) for f in categorical_features])
    categorical_preprocessor.fit(X_train, y_train)

    # X_train_preprocessed = categorical_preprocessor.transform(X_train)
    # # Append classifier to preprocessing pipeline.
    # # Now we have a full prediction pipeline.
    # clf = Pipeline(steps=[('classifier', LogisticRegression())])
    #
    # clf.fit(X_train_preprocessed, y_train)
    # X_test_preprocessed = categorical_preprocessor.transform(X_test)
    # print("model score: %.3f" % clf.score(X_test_preprocessed, y_test))

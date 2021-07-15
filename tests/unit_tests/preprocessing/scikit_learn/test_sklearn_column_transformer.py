import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scalar import \
    SKLearnPrefittedMinMaxScaler
from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_one_hot_transformer import \
    SKLearnPrefittedOneHotTransformer


def test_multiple_columns():
    dictionary = pd.DataFrame(data=[
        ["A", 1],
        ["B", 0],
        ["C", 2],
        ["D", 3],
        ["E", 4]], columns=["VALUE", "ID"])

    transformer = SKLearnPrefittedColumnTransformer(
        transformer_mapping=[
            ("a", SKLearnPrefittedMinMaxScaler(min_value=0., range_value=1.)),
            ("b", SKLearnPrefittedMinMaxScaler(min_value=0., range_value=2.)),
            ("d", SKLearnPrefittedOneHotTransformer(dictionary=dictionary))
        ]
    )
    X = pd.DataFrame(data=[
        [0.0, 1.0, 1, "A"],
        [1.0, 2.0, 2, "B"],
        [2.0, 3.0, 3, "C"],
        [3.0, 4.0, 4, "D"],
        [4.0, 5.0, 5, "E"],
    ], columns=["a", "b", "c", "d"])
    X_ = transformer.transform(X)
    print(X)
    print(X_)
    expected_X_ = np.array(
        [
            [0., 0.5, 1., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [2., 1.5, 0., 1., 0., 0.],
            [3., 2., 0., 0., 1., 0.],
            [4., 2.5, 0., 0., 0., 1.],
        ]
    )
    print(expected_X_)
    assert np.array_equal(expected_X_,X_)

def test_single_column():
    transformer = SKLearnPrefittedColumnTransformer(
        transformer_mapping=[
            ("a", SKLearnPrefittedMinMaxScaler(min_value=0., range_value=1.)),
        ]
    )
    X = pd.DataFrame(data=[
        [0.0, 1.0, 1, "A"],
        [1.0, 2.0, 2, "B"],
        [2.0, 3.0, 3, "C"],
        [3.0, 4.0, 4, "D"],
        [4.0, 5.0, 5, "E"],
    ], columns=["a", "b", "c", "d"])
    X_ = transformer.transform(X)
    print(X)
    print(X_)
    expected_X_ = np.array(
        [
            [0.],
            [1.],
            [2.],
            [3.],
            [4.],
        ]
    )
    print(expected_X_)
    assert np.array_equal(expected_X_,X_)

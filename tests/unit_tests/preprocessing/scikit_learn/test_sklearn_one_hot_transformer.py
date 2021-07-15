import timeit

import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_one_hot_transformer import \
    SKLearnPrefittedOneHotTransformer


def test_transform():
    dictionary = pd.DataFrame(data=[
        ["A", 1],
        ["B", 0],
        ["C", 2],
        ["D", 3],
        ["E", 4]], columns=["VALUE", "ID"])
    transformer = SKLearnPrefittedOneHotTransformer(dictionary)
    X = pd.DataFrame(data=[
        ["A"],
        ["C"],
        ["B"],
        ["D"]
    ], columns=["C1"])
    X_ = transformer.transform(X["C1"])
    expected_X_ = np.array(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
        ])
    print(X_)
    print(expected_X_)
    assert np.array_equal(X_, expected_X_)


def test_transform_large():
    dictionary = pd.DataFrame(data=[
        [f"{i}", i] for i in range(1000)
    ], columns=["VALUE", "ID"])
    transformer = SKLearnPrefittedOneHotTransformer(dictionary)
    X = pd.DataFrame(data=[
        [f"{i}", ] for i in range(1000)
    ], columns=["C1"])

    def run():
        X_ = transformer.transform(X["C1"])

    timer = timeit.timeit(run, number=100)
    print(timer / 100)

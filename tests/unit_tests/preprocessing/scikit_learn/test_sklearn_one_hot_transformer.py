import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_one_hot_transformer import \
    SKLearnPrefittedOneHotTransformer


def test_create_contrast_matrix():
    dictionary = pd.DataFrame(data=[
        ["A", 1],
        ["B", 0],
        ["C", 2],
        ["D", 3],
        ["E", 4]], columns=["VALUE", "ID"])
    transformer = SKLearnPrefittedOneHotTransformer(dictionary)
    contrast_matrix = transformer.create_contrast_matrix()
    expected_contrast_matrix = np.array(
        [
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    print(expected_contrast_matrix)
    print(contrast_matrix.index)
    print(contrast_matrix.values)
    print(contrast_matrix)
    assert contrast_matrix.index.to_list() == ["A","B","C","D","E"]
    assert np.array_equal(expected_contrast_matrix, contrast_matrix.values)

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


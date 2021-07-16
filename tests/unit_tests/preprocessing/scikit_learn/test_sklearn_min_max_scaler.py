import numpy as np
import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_min_max_scaler import SKLearnPrefittedMinMaxScaler


def test_happy_path():
    X = pd.DataFrame(data=[
        [0],
        [1],
        [2],
        [3],
        [4],
        [1.5]
    ])
    transformer = SKLearnPrefittedMinMaxScaler(min_value=1, range_value=2)
    X_ = transformer.transform(X[0])
    expected_X = pd.DataFrame(data=[[-0.5], [0.], [0.5], [1.], [1.5], [0.25]])[0]
    print(X_)
    print(expected_X)
    assert np.array_equal(X_, expected_X)

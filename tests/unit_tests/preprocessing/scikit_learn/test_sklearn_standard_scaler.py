import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_standard_scaler import \
    SKLearnPrefittedStandardScaler


def test_happy_path():
    X = pd.DataFrame(data=[
        [0],
        [1],
        [2],
        [3],
        [4],
        [1.5]
    ])
    s = StandardScaler().fit(np.array([1.,2.]).reshape(-1,1))
    print(s.transform(X))
    print(s.__dict__)
    stddev = pd.DataFrame([1.,2.], dtype=np.float64).std(ddof=0).values.item()
    print("stddev",stddev)
    transformer = SKLearnPrefittedStandardScaler(avg_value=1.5, stddev_value=stddev)
    X_ = transformer.transform(X[0])
    expected_X = pd.DataFrame(data=[[-3.], [-1.], [1.], [3.], [5.], [0.]])[0]
    print(X_)
    print(expected_X)
    assert np.array_equal(X_, expected_X)

import numpy as np
import pandas as pd
import pytest

from exasol_data_science_utils_python.model_utils.reservoir_shuffle import ReservoirShuffle


class TestValue:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return other.value == self.value

    def __gt__(self, other):
        return other.value > self.value

    def __repr__(self):
        return f"TestValue({self.value})"


map_to_test_value = np.vectorize(lambda y: TestValue(y))


def two_types(i, x):
    if i == 0:
        return x
    else:
        return x * 0.1


@pytest.mark.parametrize("input_size,batch_size,buffer_size,columns_count,column_generator", [
    (50, 10, 20, 1, lambda i, x: x),
    (50, 10, 20, 2, lambda i, x: x),
    (50, 20, 20, 1, lambda i, x: x),
    (50, 10, 25, 1, lambda i, x: x),
    (100, 20, 20, 1, lambda i, x: x),
    (100, 20, 20, 2, two_types),
    (100, 20, 20, 1, lambda i, x: map_to_test_value(x)),
    (100000, 1000, 50000, 10, lambda i, x: x),
])
def test_reservoir_shuffle(input_size, batch_size, columns_count, buffer_size, column_generator):
    columns = {i: np.arange(input_size) for i in range(columns_count)}
    columns = {i: column_generator(i, x) for i, x in columns.items()}
    input_df = pd.DataFrame(data=columns, index=np.arange(input_size))
    def batch_iterator(input_df, batch_size):
        for i in range(0, len(input_df), batch_size):
            yield input_df.iloc[i:i + batch_size]

    batches = list(batch_iterator(input_df, batch_size))
    elements_of_batches = []
    for batch in batches:
        elements_of_batches.extend(batch.values.tolist())
    sorted_elements_of_batches = list(sorted(elements_of_batches))

    reservoir = ReservoirShuffle(batch_iterator(input_df, batch_size), buffer_size, batch_size)
    result = list(reservoir.shuffle())
    elements_of_result = []
    for batch in result:
        elements_of_result.extend(batch.values.tolist())
    sorted_elements_of_result = list(sorted(elements_of_result))

    assert sorted_elements_of_batches == sorted_elements_of_result
    assert sorted_elements_of_batches != elements_of_result
